/*
 * libs3 implementation. See libs3.h for the API contract.
 *
 * Ported and generalised from volume-cartographer's http_fetch.hpp /
 * aws_auth.cpp: thread-local reused curl handle, exponential-backoff retry,
 * process-wide fast abort, libcurl built-in SigV4, and the full AWS
 * credential resolution chain (IMDSv2 cached in-process, SSO scan, INI,
 * env). JSON and XML are handled by purpose-built scrapers, not a general
 * parser -- they only need to read the handful of fields S3/IMDS emit.
 */
#define _GNU_SOURCE
#include "libs3.h"
#include "libs3_internal.h"

/* Scraper/parse helpers are `static` normally, but get external linkage
 * under -DLIBS3_TESTING so white-box tests can link (not #include) them
 * -- keeping a single instrumented translation unit for honest coverage. */
#ifdef LIBS3_TESTING
#  define LIBS3_INTERNAL
#else
#  define LIBS3_INTERNAL static
#endif

#include <curl/curl.h>

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>   /* ftruncate, fileno (sink rewind on retry) */
#endif
#include <time.h>

/* ================================================================== */
/* Small utilities                                                     */
/* ================================================================== */

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static char *strndup_(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* Grow-on-append byte buffer. */
typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} buf_t;

static bool buf_append(buf_t *b, const void *p, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->len + n + 1) nc *= 2;
        uint8_t *nd = realloc(b->data, nc);
        if (!nd) return false;
        b->data = nd;
        b->cap = nc;
    }
    memcpy(b->data + b->len, p, n);
    b->len += n;
    b->data[b->len] = '\0';   /* keep NUL-terminated for text convenience */
    return true;
}

/* Append formatted text to a heap C-string (realloc). Returns new ptr. */
static char *str_appendf(char *dst, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return dst;
    size_t old = dst ? strlen(dst) : 0;
    char *nd = realloc(dst, old + (size_t)need + 1);
    if (!nd) return dst;
    va_start(ap, fmt);
    vsnprintf(nd + old, (size_t)need + 1, fmt, ap);
    va_end(ap);
    return nd;
}

/* Thread-local last-error message. */
static _Thread_local char g_last_error[512];
static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof g_last_error, fmt, ap);
    va_end(ap);
}

/* ================================================================== */
/* Global curl init + process-wide abort flag                          */
/* ================================================================== */

static atomic_int  g_curl_refcount = 0;
static atomic_bool g_abort_flag    = false;
static pthread_mutex_t g_curl_init_mtx = PTHREAD_MUTEX_INITIALIZER;

static void curl_global_ref(void) {
    /* curl_global_init() is NOT thread-safe and must FINISH before any thread
     * uses curl. The old atomic-refcount-only version let a second concurrent
     * caller proceed while init was still running. Serialize under a mutex so
     * concurrent s3_client_new() (e.g. opening two volumes at once) is safe. */
    pthread_mutex_lock(&g_curl_init_mtx);
    if (g_curl_refcount++ == 0)
        curl_global_init(CURL_GLOBAL_DEFAULT);
    pthread_mutex_unlock(&g_curl_init_mtx);
}
static void curl_global_unref(void) {
    pthread_mutex_lock(&g_curl_init_mtx);
    if (--g_curl_refcount == 0)
        curl_global_cleanup();
    pthread_mutex_unlock(&g_curl_init_mtx);
}

void s3_global_abort(void)      { atomic_store(&g_abort_flag, true); }
void s3_global_reset_abort(void){ atomic_store(&g_abort_flag, false); }
bool s3_global_is_aborted(void) { return atomic_load(&g_abort_flag); }

static int xferinfo_cb(void *u, curl_off_t a, curl_off_t b,
                       curl_off_t c, curl_off_t d) {
    (void)u; (void)a; (void)b; (void)c; (void)d;
    return atomic_load(&g_abort_flag) ? 1 : 0;
}

/* Thread-local reused easy handle: preserves the connection pool and TLS
 * session IDs across calls so back-to-back S3 fetches skip the handshake. */
static pthread_key_t g_handle_key;
static pthread_once_t g_handle_once = PTHREAD_ONCE_INIT;

static void handle_destructor(void *h) {
    if (h) curl_easy_cleanup((CURL *)h);
}
static void make_handle_key(void) {
    pthread_key_create(&g_handle_key, handle_destructor);
}
static CURL *thread_handle(void) {
    pthread_once(&g_handle_once, make_handle_key);
    CURL *h = pthread_getspecific(g_handle_key);
    if (!h) {
        h = curl_easy_init();
        pthread_setspecific(g_handle_key, h);
    }
    return h;
}

/* ================================================================== */
/* Status strings                                                      */
/* ================================================================== */

const char *s3_status_str(s3_status st) {
    switch (st) {
    case S3_OK:               return "ok";
    case S3_ERR_INVALID_ARG:  return "invalid argument";
    case S3_ERR_OOM:          return "out of memory";
    case S3_ERR_CURL:         return "curl transport failure";
    case S3_ERR_HTTP:         return "non-2xx HTTP status";
    case S3_ERR_NO_CREDS:     return "no AWS credentials found";
    case S3_ERR_PARSE:        return "malformed response";
    case S3_ERR_IO:           return "local I/O failure";
    case S3_ERR_ABORTED:      return "aborted";
    }
    return "unknown error";
}

/* ================================================================== */
/* S3 URL parsing                                                      */
/* ================================================================== */

bool s3_url_is_s3(const char *url) {
    if (!url) return false;
    if (strncmp(url, "s3://", 5) == 0 || strncmp(url, "S3://", 5) == 0)
        return true;
    if ((strncmp(url, "s3+", 3) == 0 || strncmp(url, "S3+", 3) == 0) &&
        strstr(url, "://"))
        return true;
    return false;
}

s3_status s3_url_parse(const char *url, s3_url *out) {
    if (!url || !out || !s3_url_is_s3(url)) return S3_ERR_INVALID_ARG;
    memset(out, 0, sizeof *out);

    const char *sep = strstr(url, "://");
    if (!sep) return S3_ERR_INVALID_ARG;

    /* scheme may be "s3" or "s3+REGION" */
    const char *plus = memchr(url, '+', (size_t)(sep - url));
    char *region = plus ? strndup_(plus + 1, (size_t)(sep - plus - 1))
                        : xstrdup("");
    const char *rest = sep + 3;
    const char *slash = strchr(rest, '/');

    char *bucket, *key;
    if (!slash) {
        bucket = xstrdup(rest);
        key = xstrdup("");
    } else {
        bucket = strndup_(rest, (size_t)(slash - rest));
        key = xstrdup(slash + 1);
    }
    if (!region || !bucket || !key) {
        free(region); free(bucket); free(key);
        return S3_ERR_OOM;
    }
    out->bucket = bucket;
    out->key = key;
    out->region = region;
    return S3_OK;
}

void s3_url_free(s3_url *u) {
    if (!u) return;
    free(u->bucket); free(u->key); free(u->region);
    memset(u, 0, sizeof *u);
}

s3_status s3_url_to_https(const s3_url *u, char *buf, size_t buflen) {
    if (!u || !buf) return S3_ERR_INVALID_ARG;
    int n;
    if (u->region && u->region[0])
        n = snprintf(buf, buflen, "https://%s.s3.%s.amazonaws.com/%s",
                     u->bucket, u->region, u->key ? u->key : "");
    else
        n = snprintf(buf, buflen, "https://%s.s3.amazonaws.com/%s",
                     u->bucket, u->key ? u->key : "");
    if (n < 0 || (size_t)n >= buflen) return S3_ERR_INVALID_ARG;
    return S3_OK;
}

/* Render to a freshly malloc'd string (internal helper). */
static char *s3_url_to_https_alloc(const s3_url *u) {
    size_t need = strlen(u->bucket) + (u->region ? strlen(u->region) : 0) +
                  (u->key ? strlen(u->key) : 0) + 64;
    char *buf = malloc(need);
    if (!buf) return NULL;
    if (s3_url_to_https(u, buf, need) != S3_OK) { free(buf); return NULL; }
    return buf;
}

/* Forward decl: client type defined later. */
struct s3_client;

/*
 * Resolve a user-supplied URL to the concrete HTTP(S) URL to fetch,
 * honouring an optional non-AWS endpoint (path-style). Any "?query"
 * suffix (multipart/list params) is split off, the s3:// part parsed,
 * then the query re-appended. Plain http(s):// URLs pass through.
 */
static char *resolve_url_for(const char *endpoint, bool insecure,
                             const char *url) {
    if (!url) return NULL;
    if (!s3_url_is_s3(url))
        return strdup(url);

    const char *q = strchr(url, '?');
    char *base = q ? strndup_(url, (size_t)(q - url)) : strdup(url);
    if (!base) return NULL;

    s3_url u;
    if (s3_url_parse(base, &u) != S3_OK) { free(base); return NULL; }
    free(base);

    char *out;
    if (endpoint && endpoint[0]) {
        /* path-style: <scheme>://<endpoint>/<bucket>/<key> */
        out = NULL;
        out = str_appendf(out, "%s://%s/%s/%s",
                          insecure ? "http" : "https",
                          endpoint, u.bucket, u.key ? u.key : "");
    } else {
        out = s3_url_to_https_alloc(&u);
    }
    s3_url_free(&u);
    if (out && q)
        out = str_appendf(out, "%s", q);   /* re-attach ?query */
    return out;
}

/* ================================================================== */
/* Minimal JSON scraper                                                */
/*                                                                     */
/* Only used for flat IMDS / `aws configure export-credentials` objects */
/* like {"AccessKeyId":"..","SecretAccessKey":"..","Token":".."}. Finds */
/* "key" then the next "..."-delimited string value. Handles \" and \\  */
/* escapes; not a general parser (no nesting, numbers, arrays).         */
/* ================================================================== */

LIBS3_INTERNAL char *json_string_field(const char *json, const char *key) {
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strchr(p, '"')) != NULL) {
        const char *kstart = p + 1;
        if (strncmp(kstart, key, klen) == 0 && kstart[klen] == '"') {
            const char *q = kstart + klen + 1;
            while (*q && *q != ':') q++;
            if (*q != ':') return NULL;
            q++;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q != '"') return NULL;
            q++;
            /* measure with escapes */
            buf_t b = {0};
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) {
                    char c = q[1];
                    char out = c;
                    if (c == 'n') out = '\n';
                    else if (c == 't') out = '\t';
                    else if (c == 'r') out = '\r';
                    buf_append(&b, &out, 1);
                    q += 2;
                } else {
                    buf_append(&b, q, 1);
                    q++;
                }
            }
            char *res = b.data ? (char *)b.data : xstrdup("");
            return res;
        }
        p++;
    }
    return NULL;
}

/* ================================================================== */
/* Minimal XML scraper (ListObjectsV2 / multipart responses)           */
/* ================================================================== */

/* First text content of <tag>...</tag> after `from`, or NULL. Sets *end
 * to just past </tag> so callers can iterate repeated elements. */
LIBS3_INTERNAL char *xml_tag(const char *from, const char *tag, const char **end) {
    char open[64], close[64];
    snprintf(open, sizeof open, "<%s>", tag);
    snprintf(close, sizeof close, "</%s>", tag);
    const char *o = strstr(from, open);
    if (!o) return NULL;
    o += strlen(open);
    const char *c = strstr(o, close);
    if (!c) return NULL;
    if (end) *end = c + strlen(close);
    return strndup_(o, (size_t)(c - o));
}

/* Percent-decode in place (S3 list uses encoding-type=url). */
LIBS3_INTERNAL void url_decode_inplace(char *s) {
    char *w = s;
    for (char *r = s; *r; r++) {
        if (*r == '%' && isxdigit((unsigned char)r[1]) &&
            isxdigit((unsigned char)r[2])) {
            int hi = isdigit((unsigned char)r[1]) ? r[1]-'0'
                     : (tolower((unsigned char)r[1])-'a'+10);
            int lo = isdigit((unsigned char)r[2]) ? r[2]-'0'
                     : (tolower((unsigned char)r[2])-'a'+10);
            *w++ = (char)((hi << 4) | lo);
            r += 2;
        } else if (*r == '+') {
            *w++ = ' ';
        } else {
            *w++ = *r;
        }
    }
    *w = '\0';
}

/* URL-encode `s` as a query-string value into a fresh string. Only the
 * RFC 3986 unreserved set is left as-is; everything else (including '/',
 * '+', '=') is percent-encoded -- required for S3 continuation tokens,
 * which arrive un-encoded and contain '+' '/' '='. */
LIBS3_INTERNAL char *url_encode(const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    char *out = malloc(strlen(s) * 3 + 1);
    if (!out) return NULL;
    char *w = out;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (isalnum(*p) || *p=='-' || *p=='_' || *p=='.' || *p=='~') {
            *w++ = (char)*p;
        } else {
            *w++ = '%';
            *w++ = hex[*p >> 4];
            *w++ = hex[*p & 15];
        }
    }
    *w = '\0';
    return out;
}

/* ================================================================== */
/* Client                                                              */
/* ================================================================== */

struct s3_client {
    /* owned copies of all config strings */
    s3_credentials      creds;
    s3_cred_provider_fn cred_provider;
    void               *cred_userdata;
    char *region;
    char *bearer_token;
    char *basic_user;
    char *basic_pass;
    char *user_agent;
    char *endpoint;            /* non-AWS host[:port], NULL for real AWS */
    bool  endpoint_insecure;   /* http:// instead of https:// for endpoint */
    long  connect_timeout_s;
    long  transfer_timeout_s;
    int   max_retries;
    bool  follow_redirects;
};

static void dup_creds(s3_credentials *dst, const s3_credentials *src) {
    dst->access_key   = xstrdup(src ? src->access_key   : NULL);
    dst->secret_key   = xstrdup(src ? src->secret_key   : NULL);
    dst->session_token= xstrdup(src ? src->session_token: NULL);
    dst->region       = xstrdup(src ? src->region       : NULL);
}

void s3_credentials_free(s3_credentials *c) {
    if (!c) return;
    free(c->access_key); free(c->secret_key);
    free(c->session_token); free(c->region);
    memset(c, 0, sizeof *c);
}

s3_client *s3_client_new(const s3_config *cfg) {
    curl_global_ref();
    s3_client *c = calloc(1, sizeof *c);
    if (!c) { curl_global_unref(); return NULL; }

    dup_creds(&c->creds, cfg ? &cfg->creds : NULL);
    c->cred_provider = cfg ? cfg->cred_provider : NULL;
    c->cred_userdata = cfg ? cfg->cred_userdata : NULL;
    c->region       = xstrdup(cfg ? cfg->region : NULL);
    c->bearer_token = xstrdup(cfg ? cfg->bearer_token : NULL);
    c->basic_user   = xstrdup(cfg ? cfg->basic_user : NULL);
    c->basic_pass   = xstrdup(cfg ? cfg->basic_pass : NULL);
    c->user_agent   = xstrdup(cfg && cfg->user_agent ? cfg->user_agent
                                                     : "libs3/1.0");
    c->endpoint     = (cfg && cfg->endpoint && cfg->endpoint[0])
                          ? xstrdup(cfg->endpoint) : NULL;
    c->endpoint_insecure = cfg ? cfg->endpoint_insecure : false;
    c->connect_timeout_s  = cfg && cfg->connect_timeout_s  ? cfg->connect_timeout_s  : 10;
    c->transfer_timeout_s = cfg && cfg->transfer_timeout_s ? cfg->transfer_timeout_s : 30;
    c->max_retries        = cfg ? cfg->max_retries : 3;
    c->follow_redirects   = cfg ? cfg->follow_redirects : true;
    return c;
}

void s3_client_free(s3_client *c) {
    if (!c) return;
    s3_credentials_free(&c->creds);
    free(c->region); free(c->bearer_token); free(c->basic_user);
    free(c->basic_pass); free(c->user_agent); free(c->endpoint);
    free(c);
    curl_global_unref();
}

const char *s3_client_last_error(const s3_client *c) {
    (void)c;
    return g_last_error;
}

/* ================================================================== */
/* curl callbacks                                                      */
/* ================================================================== */

static size_t write_cb(char *ptr, size_t sz, size_t nm, void *ud) {
    buf_t *b = ud;
    size_t n = sz * nm;
    return buf_append(b, ptr, n) ? n : 0;
}

/* Streaming sink: write the body straight to a FILE* in constant memory
 * (used by s3_get_to_file so large objects never buffer fully in RAM). */
static size_t file_write_cb(char *ptr, size_t sz, size_t nm, void *ud) {
    return fwrite(ptr, sz, nm, (FILE *)ud) * sz;
}

/* Trim a header value (skip prefix, strip surrounding ws/CRLF) into *dst. */
static void hdr_store(char **dst, const char *line, size_t n, size_t skip) {
    const char *v = line + skip;
    size_t vn = n - skip;
    while (vn && (*v == ' ' || *v == '\t')) { v++; vn--; }
    while (vn && (v[vn-1] == '\r' || v[vn-1] == '\n' || v[vn-1] == ' ')) vn--;
    free(*dst);
    *dst = strndup_(v, vn);
}

static size_t header_cb(char *ptr, size_t sz, size_t nm, void *ud) {
    s3_response *r = ud;
    size_t n = sz * nm;
    /* case-insensitive prefix tests */
    if (n > 13 && strncasecmp(ptr, "content-type:", 13) == 0) {
        hdr_store(&r->content_type, ptr, n, 13);
    } else if (n > 15 && strncasecmp(ptr, "content-length:", 15) == 0) {
        r->content_length = strtoull(ptr + 15, NULL, 10);
    } else if (n > 5 && strncasecmp(ptr, "etag:", 5) == 0) {
        hdr_store(&r->etag, ptr, n, 5);
    } else if (n > 14 && strncasecmp(ptr, "last-modified:", 14) == 0) {
        hdr_store(&r->last_modified, ptr, n, 14);
    }
    return n;
}

/* Upload source: either an in-memory span (data != NULL) or a streamed
 * FILE* (fp != NULL) read in constant memory. */
typedef struct {
    const uint8_t *data; size_t len, off;
    FILE *fp;
} read_state;
static size_t read_cb(char *buf, size_t sz, size_t nm, void *ud) {
    read_state *s = ud;
    if (s->fp)
        return fread(buf, 1, sz * nm, s->fp);
    size_t rem = s->len - s->off;
    size_t cp = sz * nm < rem ? sz * nm : rem;
    memcpy(buf, s->data + s->off, cp);
    s->off += cp;
    return cp;
}

/* ================================================================== */
/* IMDSv2 + credential resolution (ported from aws_auth.cpp)           */
/* ================================================================== */

#define IMDS_DEFAULT_BASE "http://169.254.169.254"

/* IMDS base endpoint. Normally the link-local address; overridable via
 * $LIBS3_IMDS_BASE so a local mock can stand in for tests (and as a
 * real-world escape hatch for setups that proxy the metadata service).
 * Read once per process. */
static const char *imds_base(void) {
    const char *e = getenv("LIBS3_IMDS_BASE");
    return (e && e[0]) ? e : IMDS_DEFAULT_BASE;
}

static size_t str_write_cb(char *p, size_t sz, size_t nm, void *ud) {
    buf_t *b = ud;
    size_t n = sz * nm;
    return buf_append(b, p, n) ? n : 0;
}

/* Self-contained curl call to the link-local metadata endpoint. Never
 * SigV4-signed; tight timeouts so a non-EC2 host fails over fast. */
static char *imds_request(const char *url, bool put, const char *token) {
    CURL *curl = curl_easy_init();
    if (!curl) return NULL;
    buf_t body = {0};
    struct curl_slist *h = NULL;
    if (put) {
        h = curl_slist_append(h,
            "X-aws-ec2-metadata-token-ttl-seconds: 21600");
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
    } else if (token && token[0]) {
        char hb[600];
        snprintf(hb, sizeof hb, "X-aws-ec2-metadata-token: %s", token);
        h = curl_slist_append(h, hb);
    }
    if (h) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, h);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, str_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);   /* thread-safe timeouts */
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);
    curl_easy_setopt(curl, CURLOPT_NOPROXY, "169.254.169.254");
    CURLcode rc = curl_easy_perform(curl);
    long st = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &st);
    if (h) curl_slist_free_all(h);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK || st < 200 || st >= 300) {
        free(body.data);
        return NULL;
    }
    return body.data ? (char *)body.data : xstrdup("");
}

LIBS3_INTERNAL time_t parse_iso8601(const char *s) {
    struct tm tm = {0};
    if (sscanf(s, "%d-%d-%dT%d:%d:%dZ",
               &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
               &tm.tm_hour, &tm.tm_min, &tm.tm_sec) != 6)
        return 0;
    tm.tm_year -= 1900;
    tm.tm_mon  -= 1;
    return timegm(&tm);
}

/* In-process IMDSv2 credential cache, refreshed ~5 min before expiry. */
static pthread_mutex_t g_cred_mtx = PTHREAD_MUTEX_INITIALIZER;
static s3_credentials  g_cached_creds;
static time_t          g_cached_expiry;     /* 0 == none/no-expiry */
static bool            g_cached_valid;

static bool fetch_imds_creds(s3_credentials *out, time_t *expiry) {
    const char *base = imds_base();

    char *turl = str_appendf(NULL, "%s/latest/api/token", base);
    char *token = imds_request(turl, true, "");
    free(turl);
    if (!token || !token[0]) { free(token); return false; }

    char *rurl = str_appendf(NULL,
        "%s/latest/meta-data/iam/security-credentials/", base);
    char *role = imds_request(rurl, false, token);
    free(rurl);
    if (!role || !role[0]) { free(token); free(role); return false; }
    char *nl = strchr(role, '\n');
    if (nl) *nl = '\0';

    char *url = str_appendf(NULL,
        "%s/latest/meta-data/iam/security-credentials/%s", base, role);
    char *cj = imds_request(url, false, token);
    free(url);
    free(token); free(role);
    if (!cj || !cj[0]) { free(cj); return false; }

    char *ak = json_string_field(cj, "AccessKeyId");
    char *sk = json_string_field(cj, "SecretAccessKey");
    if (!ak || !sk) { free(cj); free(ak); free(sk); return false; }
    out->access_key    = ak;
    out->secret_key    = sk;
    out->session_token = json_string_field(cj, "Token");
    if (!out->session_token) out->session_token = xstrdup("");
    out->region = xstrdup("");
    char *exp = json_string_field(cj, "Expiration");
    *expiry = exp ? parse_iso8601(exp) : 0;
    free(exp);
    free(cj);
    return true;
}

#ifdef LIBS3_TESTING
/* Clear the in-process IMDS credential cache so tests can deterministically
 * exercise cache-miss / refresh / stale-fallback transitions. */
void libs3_test_reset_imds_cache(void) {
    pthread_mutex_lock(&g_cred_mtx);
    s3_credentials_free(&g_cached_creds);
    g_cached_expiry = 0;
    g_cached_valid = false;
    pthread_mutex_unlock(&g_cred_mtx);
}
/* Resolve instance-role creds, bypassing the cache (each call hits the
 * IMDS endpoint). Lets the mock test assert the raw fetch sequence and
 * its error paths independently of caching. */
bool libs3_test_fetch_imds(s3_credentials *out, time_t *expiry) {
    return fetch_imds_creds(out, expiry);
}
/* Drive the cached path so the cache-hit / refresh / stale-fallback
 * branches are exercised. */
bool libs3_test_cached_imds(s3_credentials *out);
#endif

static bool cached_imds_creds(s3_credentials *out) {
    pthread_mutex_lock(&g_cred_mtx);
    time_t now = time(NULL);
    if (g_cached_valid &&
        (g_cached_expiry == 0 || now + 300 < g_cached_expiry)) {
        dup_creds(out, &g_cached_creds);
        pthread_mutex_unlock(&g_cred_mtx);
        return true;
    }
    s3_credentials fresh = {0};
    time_t exp = 0;
    if (fetch_imds_creds(&fresh, &exp)) {
        s3_credentials_free(&g_cached_creds);
        g_cached_creds = fresh;     /* move */
        g_cached_expiry = exp;
        g_cached_valid = true;
        dup_creds(out, &g_cached_creds);
        pthread_mutex_unlock(&g_cred_mtx);
        return true;
    }
    /* refresh failed but a still-valid copy exists -- keep using it */
    if (g_cached_valid &&
        (g_cached_expiry == 0 || now < g_cached_expiry)) {
        dup_creds(out, &g_cached_creds);
        pthread_mutex_unlock(&g_cred_mtx);
        return true;
    }
    pthread_mutex_unlock(&g_cred_mtx);
    return false;
}

#ifdef LIBS3_TESTING
bool libs3_test_cached_imds(s3_credentials *out) {
    return cached_imds_creds(out);
}
#endif

static char *getenv_dup(const char *k) {
    const char *v = getenv(k);
    return xstrdup(v ? v : "");
}

s3_status s3_credentials_from_env(s3_credentials *out) {
    if (!out) return S3_ERR_INVALID_ARG;
    memset(out, 0, sizeof *out);
    out->access_key    = getenv_dup("AWS_ACCESS_KEY_ID");
    out->secret_key    = getenv_dup("AWS_SECRET_ACCESS_KEY");
    out->session_token = getenv_dup("AWS_SESSION_TOKEN");
    out->region        = getenv_dup("AWS_DEFAULT_REGION");
    if (!out->access_key[0] || !out->secret_key[0])
        return S3_ERR_NO_CREDS;
    return S3_OK;
}

/* `aws configure export-credentials [--profile P]` -> JSON on stdout. */
static bool try_export_creds(const char *profile, s3_credentials *a) {
    char cmd[256];
    if (profile && profile[0])
        snprintf(cmd, sizeof cmd,
                 "aws configure export-credentials --profile %s 2>/dev/null",
                 profile);
    else
        snprintf(cmd, sizeof cmd,
                 "aws configure export-credentials 2>/dev/null");
    FILE *p = popen(cmd, "r");
    if (!p) return false;
    buf_t out = {0};
    char tmp[4096];
    size_t r;
    while ((r = fread(tmp, 1, sizeof tmp, p)) > 0)
        buf_append(&out, tmp, r);
    pclose(p);
    if (!out.data) return false;
    char *ak = json_string_field((char *)out.data, "AccessKeyId");
    char *sk = json_string_field((char *)out.data, "SecretAccessKey");
    char *tok = json_string_field((char *)out.data, "SessionToken");
    free(out.data);
    if (!ak || !sk) { free(ak); free(sk); free(tok); return false; }
    free(a->access_key); free(a->secret_key); free(a->session_token);
    a->access_key = ak;
    a->secret_key = sk;
    a->session_token = tok ? tok : xstrdup("");
    return true;
}

/* Trim leading/trailing ASCII whitespace in place, return s. */
static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

/* Scan ~/.aws/config for profiles that declare SSO. */
LIBS3_INTERNAL void find_sso_profiles(char ***out, size_t *n) {
    *out = NULL; *n = 0;
    const char *home = getenv("HOME");
    if (!home) return;
    char path[512];
    snprintf(path, sizeof path, "%s/.aws/config", home);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[1024], cur[256] = "";
    bool has_sso = false;
    while (fgets(line, sizeof line, f)) {
        char *l = trim(line);
        if (!*l || *l == '#' || *l == ';') continue;
        if (*l == '[') {
            if (cur[0] && has_sso) {
                *out = realloc(*out, (*n + 1) * sizeof **out);
                (*out)[(*n)++] = xstrdup(cur);
            }
            char *e = strchr(l, ']');
            if (e) *e = '\0';
            const char *name = l + 1;
            if (strncmp(name, "profile ", 8) == 0) name += 8;
            snprintf(cur, sizeof cur, "%s", name);
            has_sso = false;
            continue;
        }
        char *eq = strchr(l, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = trim(l);
        if (strcmp(key, "sso_account_id") == 0 ||
            strcmp(key, "sso_session") == 0)
            has_sso = true;
    }
    if (cur[0] && has_sso) {
        *out = realloc(*out, (*n + 1) * sizeof **out);
        (*out)[(*n)++] = xstrdup(cur);
    }
    fclose(f);
}

/* Parse an INI file, filling any requested keys that are still empty,
 * but only within the [profile] / [profile NAME] section. */
static void parse_ini(const char *path, const char *profile,
                      const char *const *keys, char **dsts, size_t nkeys) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[1024];
    bool in = false;
    while (fgets(line, sizeof line, f)) {
        char *l = trim(line);
        if (!*l || *l == '#' || *l == ';') continue;
        if (*l == '[') {
            char *e = strchr(l, ']');
            if (e) *e = '\0';
            const char *sec = l + 1;
            in = (strcmp(sec, profile) == 0) ||
                 (strncmp(sec, "profile ", 8) == 0 &&
                  strcmp(sec + 8, profile) == 0);
            continue;
        }
        if (!in) continue;
        char *eq = strchr(l, '=');
        if (!eq) continue;
        *eq = '\0';
        char *k = trim(l);
        char *v = trim(eq + 1);
        for (size_t i = 0; i < nkeys; i++)
            if (strcmp(k, keys[i]) == 0 && (!dsts[i] || !dsts[i][0])) {
                free(dsts[i]);
                dsts[i] = xstrdup(v);
            }
    }
    fclose(f);
}

s3_status s3_credentials_load(const char *profile, s3_credentials *out) {
    if (!out) return S3_ERR_INVALID_ARG;
    memset(out, 0, sizeof *out);
    out->access_key = xstrdup("");
    out->secret_key = xstrdup("");
    out->session_token = xstrdup("");
    out->region = xstrdup("");

    const char *env_profile = getenv("AWS_PROFILE");
    const char *eff_profile = (profile && profile[0]) ? profile : "default";
    bool got = false;

    /* 1. explicit profile (env or arg) */
    if (env_profile && env_profile[0])
        got = try_export_creds(env_profile, out);
    else if (strcmp(eff_profile, "default") != 0)
        got = try_export_creds(eff_profile, out);

    /* 2. EC2 instance role via cached IMDSv2 */
    if (!got) {
        s3_credentials imds = {0};
        if (cached_imds_creds(&imds)) {
            s3_credentials_free(out);
            *out = imds;
            got = true;
        }
    }

    /* 3. SSO profiles from ~/.aws/config */
    if (!got) {
        char **profs = NULL; size_t np = 0;
        find_sso_profiles(&profs, &np);
        for (size_t i = 0; i < np && !got; i++)
            got = try_export_creds(profs[i], out);
        for (size_t i = 0; i < np; i++) free(profs[i]);
        free(profs);
    }

    /* 4. default export-credentials. This is the last step that uses
       `got`; the INI/env fallbacks below test out->access_key directly,
       so we deliberately don't re-store the result. */
    if (!got)
        (void)try_export_creds("", out);

    /* 5. INI files */
    {
        const char *home = getenv("HOME");
        const char *p = (env_profile && env_profile[0]) ? env_profile
                                                        : eff_profile;
        if (home && (!out->access_key[0] || !out->secret_key[0])) {
            char path[512];
            snprintf(path, sizeof path, "%s/.aws/credentials", home);
            const char *keys[] = {"aws_access_key_id",
                                   "aws_secret_access_key",
                                   "aws_session_token"};
            char *d2[] = {NULL, NULL, NULL};
            parse_ini(path, p, keys, d2, 3);
            if (d2[0]) { free(out->access_key);    out->access_key    = d2[0]; }
            if (d2[1]) { free(out->secret_key);    out->secret_key    = d2[1]; }
            if (d2[2]) { free(out->session_token); out->session_token = d2[2]; }
        }
        if (home && !out->region[0]) {
            char path[512];
            snprintf(path, sizeof path, "%s/.aws/config", home);
            const char *keys[] = {"region"};
            char *d2[] = {NULL};
            parse_ini(path, p, keys, d2, 1);
            if (d2[0]) { free(out->region); out->region = d2[0]; }
        }
    }

    /* 6. environment variables */
    if (!out->access_key[0]) {
        free(out->access_key);  out->access_key  = getenv_dup("AWS_ACCESS_KEY_ID");
    }
    if (!out->secret_key[0]) {
        free(out->secret_key);  out->secret_key  = getenv_dup("AWS_SECRET_ACCESS_KEY");
    }
    if (!out->session_token[0]) {
        free(out->session_token); out->session_token = getenv_dup("AWS_SESSION_TOKEN");
    }
    if (!out->region[0]) {
        free(out->region);      out->region      = getenv_dup("AWS_DEFAULT_REGION");
    }

    if (!out->access_key[0] || !out->secret_key[0]) {
        set_error("credential resolution found no usable AWS keys");
        return S3_ERR_NO_CREDS;
    }
    return S3_OK;
}

/* ================================================================== */
/* Request core: auth + retry + abort                                  */
/* ================================================================== */

typedef enum { M_GET, M_GET_RANGE, M_HEAD, M_PUT, M_DELETE, M_POST } method_t;

/* Resolve the credentials to use for this request (provider or static). */
static s3_status resolve_request_creds(s3_client *c, s3_credentials *out) {
    if (c->cred_provider) {
        s3_status rc = c->cred_provider(c->cred_userdata, out);
        if (rc != S3_OK) return rc;
    } else {
        dup_creds(out, &c->creds);
    }
    if ((!out->region || !out->region[0]) && c->region && c->region[0]) {
        free(out->region);
        out->region = xstrdup(c->region);
    }
    return S3_OK;
}

/* Apply auth onto the handle. *hdrs accumulates request-scoped headers the
 * caller must curl_slist_free_all after perform. */
static void apply_auth(s3_client *c, CURL *curl, const s3_credentials *cr,
                       struct curl_slist **hdrs,
                       char *sigv4_buf, char *userpwd_buf) {
    bool have_aws = cr && cr->access_key && cr->access_key[0] &&
                    cr->secret_key && cr->secret_key[0];
    if (have_aws) {
        const char *region = (cr->region && cr->region[0]) ? cr->region
                                                           : "us-east-1";
        sprintf(sigv4_buf, "aws:amz:%s:s3", region);
        curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4_buf);
        sprintf(userpwd_buf, "%s:%s", cr->access_key, cr->secret_key);
        curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd_buf);
        if (cr->session_token && cr->session_token[0]) {
            char *h = str_appendf(NULL, "x-amz-security-token: %s",
                                  cr->session_token);
            *hdrs = curl_slist_append(*hdrs, h);
            free(h);
        }
    } else if (c->bearer_token && c->bearer_token[0]) {
        char *h = str_appendf(NULL, "Authorization: Bearer %s",
                              c->bearer_token);
        *hdrs = curl_slist_append(*hdrs, h);
        free(h);
    } else if (c->basic_user && c->basic_user[0]) {
        sprintf(userpwd_buf, "%s:%s", c->basic_user,
                c->basic_pass ? c->basic_pass : "");
        curl_easy_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_BASIC);
        curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd_buf);
    }
}

static void backoff(int attempt) {
    long ms = 200L * (1L << attempt) + (rand() % 101);
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/*
 * Perform one request with retry. `extra_hdrs` are caller-supplied headers
 * (e.g. x-amz-copy-source); ownership stays with caller. Fills *resp.
 */
static s3_status do_request(s3_client *c, const char *url, method_t method,
                            const void *body, size_t body_len,
                            FILE *body_file, long body_file_size,
                            const char *content_type,
                            const char *range,
                            const struct curl_slist *extra_hdrs,
                            FILE *sink,        /* if set, stream body here (no RAM buffer) */
                            s3_response *resp) {
    if (!c || !url || !resp) return S3_ERR_INVALID_ARG;
    memset(resp, 0, sizeof *resp);

    /* s3:// -> concrete http(s):// (honours endpoint override) */
    char *resolved = resolve_url_for(c->endpoint, c->endpoint_insecure, url);
    if (!resolved) return S3_ERR_INVALID_ARG;

    s3_status final = S3_OK;
    for (int attempt = 0; attempt <= c->max_retries; attempt++) {
        if (s3_global_is_aborted()) { final = S3_ERR_ABORTED; break; }

        s3_response_free(resp);
        memset(resp, 0, sizeof *resp);
        buf_t bbuf = {0};

        CURL *curl = thread_handle();
        if (!curl) { free(resolved); return S3_ERR_CURL; }
        curl_easy_reset(curl);

        curl_easy_setopt(curl, CURLOPT_URL, resolved);
        /* NOSIGNAL is required for thread safety: without it libcurl uses
         * SIGALRM + sigsuspend for timeouts, which races/hangs when multiple
         * threads run transfers concurrently. */
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, c->connect_timeout_s);
        /* Abort on STALL, not on size: CURLOPT_TIMEOUT is a hard cap on the whole
         * transfer, which kills large (>1GB) shard downloads that simply take a
         * while — they'd time out mid-transfer and restart forever. Use a low-
         * speed watchdog instead: abort only if throughput stays under
         * LOW_SPEED_LIMIT bytes/s for LOW_SPEED_TIME seconds (a dead/stalled
         * connection), letting a slow-but-progressing big download finish. */
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);             /* 1 KB/s */
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, c->transfer_timeout_s);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 30L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 30L);
        if (c->follow_redirects) {
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
        }
        curl_easy_setopt(curl, CURLOPT_USERAGENT, c->user_agent);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo_cb);
        if (sink) {
            rewind(sink);                 /* fresh write each retry */
#if defined(__unix__) || defined(__APPLE__)
            if (ftruncate(fileno(sink), 0) != 0) { /* best-effort: rewind already
                reset the offset; a non-regular sink (pipe) just can't truncate */ }
#endif
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, file_write_cb);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, sink);
        } else {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &bbuf);
        }
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_cb);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, resp);

        /* per-request credentials + auth */
        s3_credentials cr = {0};
        s3_status crc = resolve_request_creds(c, &cr);
        if (crc != S3_OK) { free(resolved); s3_credentials_free(&cr); return crc; }

        struct curl_slist *hdrs = NULL;
        char sigv4_buf[128], userpwd_buf[2048];
        sigv4_buf[0] = userpwd_buf[0] = '\0';
        apply_auth(c, curl, &cr, &hdrs, sigv4_buf, userpwd_buf);
        s3_credentials_free(&cr);

        /* merge caller-supplied headers */
        for (const struct curl_slist *p = extra_hdrs; p; p = p->next)
            hdrs = curl_slist_append(hdrs, p->data);

        read_state rs = {0};
        char *ct_hdr = NULL;
        switch (method) {
        case M_GET:
            curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
            break;
        case M_GET_RANGE:
            curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
            curl_easy_setopt(curl, CURLOPT_RANGE, range);
            break;
        case M_HEAD:
            curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
            break;
        case M_DELETE:
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
            break;
        case M_PUT:
        case M_POST:
            if (method == M_POST)
                curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
            else
                curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
            {
            curl_off_t clen = body_file ? (curl_off_t)body_file_size
                                        : (curl_off_t)body_len;
            curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, clen);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, clen);
            }
            if (body_file) {
                rewind(body_file);          /* fresh read each retry */
                rs.fp = body_file;
            } else {
                rs.data = body; rs.len = body_len; rs.off = 0;
            }
            curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_cb);
            curl_easy_setopt(curl, CURLOPT_READDATA, &rs);
            if (method == M_POST) {
                curl_easy_setopt(curl, CURLOPT_POST, 1L);
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, NULL);
            }
            ct_hdr = str_appendf(NULL, "Content-Type: %s",
                                 content_type ? content_type
                                              : "application/octet-stream");
            hdrs = curl_slist_append(hdrs, ct_hdr);
            break;
        }
        if (hdrs) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);

        CURLcode code = curl_easy_perform(curl);
        free(ct_hdr);
        if (hdrs) curl_slist_free_all(hdrs);

        if (code == CURLE_OK) {
            if (sink) fflush(sink);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp->status);
            resp->body = bbuf.data;       /* NULL when streamed to sink */
            resp->body_len = bbuf.len;
            bool retryable = resp->status >= 500 ||
                             resp->status == 401 || resp->status == 403;
            if (retryable && attempt < c->max_retries) {
                backoff(attempt);
                continue;
            }
            /* 304 Not Modified is a successful conditional-GET outcome,
               not an error -- the caller keeps its cached copy. */
            final = (s3_response_ok(resp) || resp->status == 304)
                        ? S3_OK : S3_ERR_HTTP;
            break;
        }

        /* transport error */
        free(bbuf.data);
        if (s3_global_is_aborted()) { final = S3_ERR_ABORTED; break; }
        if (attempt < c->max_retries) { backoff(attempt); continue; }
        set_error("curl: %s", curl_easy_strerror(code));
        final = S3_ERR_CURL;
        break;
    }

    free(resolved);
    return final;
}

void s3_response_free(s3_response *r) {
    if (!r) return;
    free(r->body);
    free(r->content_type);
    free(r->etag);
    free(r->last_modified);
    memset(r, 0, sizeof *r);
}

/* ================================================================== */
/* Object operations                                                   */
/* ================================================================== */

s3_status s3_get(s3_client *c, const char *url, s3_response *resp) {
    return do_request(c, url, M_GET, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, resp);
}

s3_status s3_get_to_file(s3_client *c, const char *url, FILE *sink, s3_response *resp) {
    if (!sink) return S3_ERR_INVALID_ARG;
    /* Streams the body straight to `sink` (constant memory); resp->body stays
     * NULL. resp->status/headers are still filled. */
    return do_request(c, url, M_GET, NULL, 0, NULL, 0, NULL, NULL, NULL, sink, resp);
}

s3_status s3_get_range(s3_client *c, const char *url,
                       uint64_t offset, uint64_t length, s3_response *resp) {
    if (length == 0) {
        if (resp) memset(resp, 0, sizeof *resp);
        return S3_OK;
    }
    char range[64];
    snprintf(range, sizeof range, "%llu-%llu",
             (unsigned long long)offset,
             (unsigned long long)(offset + length - 1));
    return do_request(c, url, M_GET_RANGE, NULL, 0, NULL, 0, NULL, range, NULL, NULL, resp);
}

s3_status s3_get_conditional(s3_client *c, const char *url,
                             const char *if_none_match, s3_response *resp) {
    if (!if_none_match || !if_none_match[0])
        return do_request(c, url, M_GET, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, resp);
    char *hv = str_appendf(NULL, "If-None-Match: %s", if_none_match);
    if (!hv) return S3_ERR_OOM;
    struct curl_slist *h = curl_slist_append(NULL, hv);
    free(hv);
    s3_status rc = do_request(c, url, M_GET, NULL, 0, NULL, 0, NULL, NULL, h, NULL, resp);
    curl_slist_free_all(h);
    return rc;
}

s3_status s3_head(s3_client *c, const char *url, s3_response *resp) {
    return do_request(c, url, M_HEAD, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, resp);
}

s3_status s3_put(s3_client *c, const char *url,
                 const void *data, size_t len,
                 const char *content_type, s3_response *resp) {
    return do_request(c, url, M_PUT, data, len, NULL, 0, content_type, NULL, NULL,
                      NULL, resp);
}

s3_status s3_put_if_match(s3_client *c, const char *url,
                          const void *data, size_t len,
                          const char *content_type,
                          const char *if_match, s3_response *resp) {
    if (!if_match || !if_match[0])
        return do_request(c, url, M_PUT, data, len, NULL, 0,
                          content_type, NULL, NULL, NULL, resp);
    char *hv = str_appendf(NULL, "If-Match: %s", if_match);
    if (!hv) return S3_ERR_OOM;
    struct curl_slist *h = curl_slist_append(NULL, hv);
    free(hv);
    s3_status rc = do_request(c, url, M_PUT, data, len, NULL, 0,
                              content_type, NULL, h, NULL, resp);
    curl_slist_free_all(h);
    return rc;
}

s3_status s3_put_file(s3_client *c, const char *url, const char *path,
                      const char *content_type, s3_response *resp) {
    if (!path) return S3_ERR_INVALID_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) { set_error("cannot open %s: %s", path, strerror(errno));
              return S3_ERR_IO; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return S3_ERR_IO; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return S3_ERR_IO; }
    rewind(f);
    /* Stream from the FILE* in constant memory (do_request rewinds on
       each retry attempt). */
    s3_status rc = do_request(c, url, M_PUT, NULL, 0, f, sz,
                              content_type, NULL, NULL, NULL, resp);
    fclose(f);
    return rc;
}

/* ================================================================== */
/* Batched ranged GET (curl_multi)                                     */
/* ================================================================== */

typedef struct {
    CURL              *eh;
    buf_t              body;
    s3_credentials     cr;
    struct curl_slist *hdrs;
    char              *url;          /* resolved, owned */
    char               range[64];
    char               sigv4[128];
    char               userpwd[2048];
    bool               in_flight;
} batch_slot;

s3_status s3_get_batch(s3_client *c,
                       const s3_range_req *reqs, size_t n,
                       size_t max_concurrency,
                       s3_response *out) {
    if (!c || (!reqs && n) || (!out && n)) return S3_ERR_INVALID_ARG;
    if (n == 0) return S3_OK;
    if (max_concurrency == 0) max_concurrency = 16;
    if (max_concurrency > n)  max_concurrency = n;
    if (s3_global_is_aborted()) return S3_ERR_ABORTED;

    batch_slot *slots = calloc(n, sizeof *slots);
    if (!slots) return S3_ERR_OOM;
    for (size_t i = 0; i < n; i++) memset(&out[i], 0, sizeof out[i]);

    CURLM *multi = curl_multi_init();
    if (!multi) { free(slots); return S3_ERR_CURL; }
    curl_multi_setopt(multi, CURLMOPT_MAX_TOTAL_CONNECTIONS,
                      (long)max_concurrency);

    s3_status final = S3_OK;
    size_t added = 0, completed = 0;

    /* Prime up to max_concurrency transfers; the loop tops it up. */
    while (completed < n && !s3_global_is_aborted()) {
        while (added < n &&
               (added - completed) < max_concurrency) {
            batch_slot *s = &slots[added];
            s->eh = curl_easy_init();
            if (!s->eh) { final = S3_ERR_CURL; goto drain; }
            s->url = resolve_url_for(c->endpoint, c->endpoint_insecure,
                                     reqs[added].url);
            if (!s->url) { final = S3_ERR_INVALID_ARG; goto drain; }

            curl_easy_setopt(s->eh, CURLOPT_URL, s->url);
            curl_easy_setopt(s->eh, CURLOPT_CONNECTTIMEOUT,
                             c->connect_timeout_s);
            curl_easy_setopt(s->eh, CURLOPT_TIMEOUT, c->transfer_timeout_s);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPIDLE, 30L);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPINTVL, 30L);
            if (c->follow_redirects) {
                curl_easy_setopt(s->eh, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(s->eh, CURLOPT_MAXREDIRS, 10L);
            }
            curl_easy_setopt(s->eh, CURLOPT_USERAGENT, c->user_agent);
            curl_easy_setopt(s->eh, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(s->eh, CURLOPT_XFERINFOFUNCTION, xferinfo_cb);
            curl_easy_setopt(s->eh, CURLOPT_HTTPGET, 1L);
            curl_easy_setopt(s->eh, CURLOPT_WRITEFUNCTION, write_cb);
            curl_easy_setopt(s->eh, CURLOPT_WRITEDATA, &s->body);
            curl_easy_setopt(s->eh, CURLOPT_HEADERFUNCTION, header_cb);
            curl_easy_setopt(s->eh, CURLOPT_HEADERDATA, &out[added]);

            if (reqs[added].length > 0) {
                snprintf(s->range, sizeof s->range, "%llu-%llu",
                    (unsigned long long)reqs[added].offset,
                    (unsigned long long)(reqs[added].offset +
                                         reqs[added].length - 1));
                curl_easy_setopt(s->eh, CURLOPT_RANGE, s->range);
            }

            s3_status crc = resolve_request_creds(c, &s->cr);
            if (crc != S3_OK) { final = crc; goto drain; }
            apply_auth(c, s->eh, &s->cr, &s->hdrs,
                       s->sigv4, s->userpwd);
            if (s->hdrs)
                curl_easy_setopt(s->eh, CURLOPT_HTTPHEADER, s->hdrs);

            curl_easy_setopt(s->eh, CURLOPT_PRIVATE, s);
            curl_multi_add_handle(multi, s->eh);
            s->in_flight = true;
            added++;
        }

        int running = 0;
        CURLMcode mc = curl_multi_perform(multi, &running);
        if (mc != CURLM_OK) { final = S3_ERR_CURL; goto drain; }

        if (running)
            curl_multi_poll(multi, NULL, 0, 200, NULL);

        CURLMsg *m;
        int inq = 0;
        while ((m = curl_multi_info_read(multi, &inq))) {
            if (m->msg != CURLMSG_DONE) continue;
            batch_slot *s = NULL;
            curl_easy_getinfo(m->easy_handle, CURLINFO_PRIVATE, &s);
            ptrdiff_t idx = s - slots;
            if (m->data.result == CURLE_OK) {
                curl_easy_getinfo(s->eh, CURLINFO_RESPONSE_CODE,
                                  &out[idx].status);
                out[idx].body = s->body.data;
                out[idx].body_len = s->body.len;
                s->body.data = NULL;
                if (!s3_response_ok(&out[idx]) &&
                    out[idx].status != 206 && out[idx].status != 304)
                    final = S3_ERR_HTTP;
            } else {
                set_error("batch[%td]: %s", idx,
                          curl_easy_strerror(m->data.result));
                final = S3_ERR_CURL;
            }
            curl_multi_remove_handle(multi, s->eh);
            curl_easy_cleanup(s->eh);
            s->eh = NULL;
            s->in_flight = false;
            completed++;
        }
    }
    if (s3_global_is_aborted()) final = S3_ERR_ABORTED;

drain:
    for (size_t i = 0; i < n; i++) {
        batch_slot *s = &slots[i];
        if (s->in_flight && s->eh) curl_multi_remove_handle(multi, s->eh);
        if (s->eh) curl_easy_cleanup(s->eh);
        if (s->hdrs) curl_slist_free_all(s->hdrs);
        s3_credentials_free(&s->cr);
        free(s->url);
        free(s->body.data);   /* NULL unless transfer never completed */
    }
    curl_multi_cleanup(multi);
    free(slots);
    return final;
}

s3_status s3_delete(s3_client *c, const char *url, s3_response *resp) {
    return do_request(c, url, M_DELETE, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, resp);
}

s3_status s3_copy(s3_client *c, const char *src_url, const char *dst_url,
                  s3_response *resp) {
    if (!src_url || !dst_url) return S3_ERR_INVALID_ARG;
    s3_url s;
    if (s3_url_parse(src_url, &s) != S3_OK) return S3_ERR_INVALID_ARG;
    /* x-amz-copy-source: /bucket/key */
    char *hv = str_appendf(NULL, "x-amz-copy-source: /%s/%s",
                           s.bucket, s.key ? s.key : "");
    s3_url_free(&s);
    struct curl_slist *h = curl_slist_append(NULL, hv);
    free(hv);
    s3_status rc = do_request(c, dst_url, M_PUT, "", 0, NULL, 0, NULL, NULL, h, NULL, resp);
    curl_slist_free_all(h);
    return rc;
}

/* ================================================================== */
/* Multipart upload                                                    */
/* ================================================================== */

struct s3_multipart {
    s3_client *client;
    char *url;          /* original s3:// or https:// target */
    char *upload_id;
    char *content_type;
    /* completed parts, in order */
    int   *part_numbers;
    char **etags;
    size_t part_count, part_cap;
};

s3_status s3_multipart_create(s3_client *c, const char *url,
                              const char *content_type, s3_multipart **out) {
    if (!c || !url || !out) return S3_ERR_INVALID_ARG;
    *out = NULL;

    /* POST <url>?uploads */
    char *u = str_appendf(NULL, "%s?uploads", url);
    s3_response r = {0};
    s3_status rc = do_request(c, u, M_POST, "", 0, NULL, 0, content_type, NULL, NULL,
                              NULL, &r);
    free(u);
    if (rc != S3_OK) { s3_response_free(&r); return rc; }

    char *uid = r.body ? xml_tag((char *)r.body, "UploadId", NULL) : NULL;
    s3_response_free(&r);
    if (!uid) { set_error("multipart: no UploadId in response");
                return S3_ERR_PARSE; }

    s3_multipart *m = calloc(1, sizeof *m);
    if (!m) { free(uid); return S3_ERR_OOM; }
    m->client = c;
    m->url = xstrdup(url);
    m->upload_id = uid;
    m->content_type = xstrdup(content_type ? content_type : "");
    *out = m;
    return S3_OK;
}

/* Record a completed part's number + ETag for s3_multipart_complete.
 * Takes ownership of `etag`. Single-threaded callers only (the parallel
 * uploader drives curl_multi from one thread, so no locking needed). */
static void mp_register_part(s3_multipart *m, int pn, char *etag) {
    if (m->part_count == m->part_cap) {
        size_t nc = m->part_cap ? m->part_cap * 2 : 8;
        m->part_numbers = realloc(m->part_numbers, nc * sizeof(int));
        m->etags = realloc(m->etags, nc * sizeof(char *));
        m->part_cap = nc;
    }
    m->part_numbers[m->part_count] = pn;
    m->etags[m->part_count] = etag;
    m->part_count++;
}

static s3_status mp_put_part(s3_multipart *m, int pn,
                             const void *data, size_t len,
                             FILE *fp, long fp_size) {
    if (!m || pn < 1) return S3_ERR_INVALID_ARG;
    char *u = str_appendf(NULL, "%s?partNumber=%d&uploadId=%s",
                          m->url, pn, m->upload_id);
    s3_response r = {0};
    s3_status rc = do_request(m->client, u, M_PUT, data, len, fp, fp_size,
                              NULL, NULL, NULL, NULL, &r);
    free(u);
    if (rc != S3_OK) { s3_response_free(&r); return rc; }

    /* S3 returns the part's ETag in the response header; completion needs
     * the exact value (quotes included) echoed back per part. */
    if (!r.etag || !r.etag[0]) {
        set_error("multipart: part %d response had no ETag", pn);
        s3_response_free(&r);
        return S3_ERR_PARSE;
    }
    char *etag = xstrdup(r.etag);
    s3_response_free(&r);
    mp_register_part(m, pn, etag);
    return S3_OK;
}

s3_status s3_multipart_upload_part(s3_multipart *m, int part_number,
                                   const void *data, size_t len) {
    return mp_put_part(m, part_number, data, len, NULL, 0);
}

s3_status s3_multipart_upload_part_file(s3_multipart *m, int part_number,
                                        const char *path) {
    if (!path) return S3_ERR_INVALID_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) return S3_ERR_IO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return S3_ERR_IO; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return S3_ERR_IO; }
    rewind(f);
    /* Stream the part directly from disk -- constant memory regardless
       of part size (matters for multi-GB recompressed zarr volumes). */
    s3_status rc = mp_put_part(m, part_number, NULL, 0, f, sz);
    fclose(f);
    return rc;
}

/* ---- parallel multipart upload (curl_multi) ----------------------- */

typedef struct {
    CURL              *eh;
    s3_response        resp;        /* captures the part ETag header */
    s3_credentials     cr;
    struct curl_slist *hdrs;
    char              *url;         /* resolved ?partNumber=&uploadId= */
    FILE              *fp;          /* file source (NULL if buffer) */
    read_state         rs;
    char               sigv4[128];
    char               userpwd[2048];
    int                part_number;
    bool               in_flight;
    bool               done_ok;
} mp_slot;

s3_status s3_multipart_upload_parts_parallel(s3_multipart *m,
                                             const s3_part_src *parts,
                                             size_t n,
                                             size_t max_concurrency) {
    if (!m || (!parts && n)) return S3_ERR_INVALID_ARG;
    if (n == 0) return S3_OK;
    if (max_concurrency == 0) max_concurrency = 8;
    if (max_concurrency > n)  max_concurrency = n;
    if (s3_global_is_aborted()) return S3_ERR_ABORTED;

    /* validate inputs up front */
    for (size_t i = 0; i < n; i++) {
        if (parts[i].part_number < 1) return S3_ERR_INVALID_ARG;
        bool buf = parts[i].data != NULL;
        bool file = parts[i].path != NULL;
        if (buf == file) return S3_ERR_INVALID_ARG;  /* exactly one */
    }

    mp_slot *slots = calloc(n, sizeof *slots);
    if (!slots) return S3_ERR_OOM;

    CURLM *multi = curl_multi_init();
    if (!multi) { free(slots); return S3_ERR_OOM; }
    curl_multi_setopt(multi, CURLMOPT_MAX_TOTAL_CONNECTIONS,
                      (long)max_concurrency);

    s3_status final = S3_OK;
    size_t added = 0, completed = 0;

    while (completed < n && !s3_global_is_aborted()) {
        while (added < n && (added - completed) < max_concurrency) {
            mp_slot *s = &slots[added];
            const s3_part_src *p = &parts[added];
            s->part_number = p->part_number;

            s->eh = curl_easy_init();
            if (!s->eh) { final = S3_ERR_CURL; goto drain; }

            char *base = str_appendf(NULL,
                "%s?partNumber=%d&uploadId=%s",
                m->url, p->part_number, m->upload_id);
            s->url = resolve_url_for(m->client->endpoint,
                                     m->client->endpoint_insecure, base);
            free(base);
            if (!s->url) { final = S3_ERR_INVALID_ARG; goto drain; }

            curl_off_t clen;
            if (p->path) {
                s->fp = fopen(p->path, "rb");
                if (!s->fp) { final = S3_ERR_IO; goto drain; }
                if (fseek(s->fp, 0, SEEK_END) != 0) {
                    final = S3_ERR_IO; goto drain; }
                long sz = ftell(s->fp);
                if (sz < 0) { final = S3_ERR_IO; goto drain; }
                rewind(s->fp);
                clen = (curl_off_t)sz;
                s->rs.fp = s->fp;
            } else {
                s->rs.data = p->data;
                s->rs.len = p->len;
                s->rs.off = 0;
                clen = (curl_off_t)p->len;
            }

            curl_easy_setopt(s->eh, CURLOPT_URL, s->url);
            curl_easy_setopt(s->eh, CURLOPT_CONNECTTIMEOUT,
                             m->client->connect_timeout_s);
            curl_easy_setopt(s->eh, CURLOPT_TIMEOUT,
                             m->client->transfer_timeout_s);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPIDLE, 30L);
            curl_easy_setopt(s->eh, CURLOPT_TCP_KEEPINTVL, 30L);
            if (m->client->follow_redirects) {
                curl_easy_setopt(s->eh, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(s->eh, CURLOPT_MAXREDIRS, 10L);
            }
            curl_easy_setopt(s->eh, CURLOPT_USERAGENT,
                             m->client->user_agent);
            curl_easy_setopt(s->eh, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(s->eh, CURLOPT_XFERINFOFUNCTION, xferinfo_cb);
            curl_easy_setopt(s->eh, CURLOPT_UPLOAD, 1L);
            curl_easy_setopt(s->eh, CURLOPT_INFILESIZE_LARGE, clen);
            curl_easy_setopt(s->eh, CURLOPT_READFUNCTION, read_cb);
            curl_easy_setopt(s->eh, CURLOPT_READDATA, &s->rs);
            curl_easy_setopt(s->eh, CURLOPT_HEADERFUNCTION, header_cb);
            curl_easy_setopt(s->eh, CURLOPT_HEADERDATA, &s->resp);

            s3_status crc = resolve_request_creds(m->client, &s->cr);
            if (crc != S3_OK) { final = crc; goto drain; }
            apply_auth(m->client, s->eh, &s->cr, &s->hdrs,
                       s->sigv4, s->userpwd);
            if (s->hdrs)
                curl_easy_setopt(s->eh, CURLOPT_HTTPHEADER, s->hdrs);

            curl_easy_setopt(s->eh, CURLOPT_PRIVATE, s);
            curl_multi_add_handle(multi, s->eh);
            s->in_flight = true;
            added++;
        }

        int running = 0;
        if (curl_multi_perform(multi, &running) != CURLM_OK) {
            final = S3_ERR_CURL; goto drain;
        }
        if (running)
            curl_multi_poll(multi, NULL, 0, 200, NULL);

        CURLMsg *msg;
        int inq = 0;
        while ((msg = curl_multi_info_read(multi, &inq))) {
            if (msg->msg != CURLMSG_DONE) continue;
            mp_slot *s = NULL;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &s);
            if (msg->data.result == CURLE_OK) {
                curl_easy_getinfo(s->eh, CURLINFO_RESPONSE_CODE,
                                  &s->resp.status);
                if (s3_response_ok(&s->resp) &&
                    s->resp.etag && s->resp.etag[0]) {
                    mp_register_part(m, s->part_number,
                                     xstrdup(s->resp.etag));
                    s->done_ok = true;
                } else {
                    set_error("multipart part %d: status %ld%s",
                              s->part_number, s->resp.status,
                              s->resp.etag ? "" : " (no ETag)");
                    final = S3_ERR_HTTP;
                }
            } else {
                set_error("multipart part %d: %s", s->part_number,
                          curl_easy_strerror(msg->data.result));
                final = S3_ERR_CURL;
            }
            curl_multi_remove_handle(multi, s->eh);
            curl_easy_cleanup(s->eh);
            s->eh = NULL;
            s->in_flight = false;
            if (s->fp) { fclose(s->fp); s->fp = NULL; }
            completed++;
        }
    }
    if (s3_global_is_aborted()) final = S3_ERR_ABORTED;

drain:
    for (size_t i = 0; i < n; i++) {
        mp_slot *s = &slots[i];
        if (s->in_flight && s->eh) curl_multi_remove_handle(multi, s->eh);
        if (s->eh) curl_easy_cleanup(s->eh);
        if (s->hdrs) curl_slist_free_all(s->hdrs);
        s3_credentials_free(&s->cr);
        s3_response_free(&s->resp);
        if (s->fp) fclose(s->fp);
        free(s->url);
    }
    curl_multi_cleanup(multi);
    free(slots);
    return final;
}

static void mp_free(s3_multipart *m) {
    if (!m) return;
    free(m->url); free(m->upload_id); free(m->content_type);
    for (size_t i = 0; i < m->part_count; i++) free(m->etags[i]);
    free(m->etags); free(m->part_numbers);
    free(m);
}

s3_status s3_multipart_complete(s3_multipart *m, s3_response *resp) {
    if (!m) return S3_ERR_INVALID_ARG;

    /* S3 requires the Part list sorted ascending by PartNumber. Parts may
       have been registered out of order -- always with the parallel
       uploader (completion order != part order), and legitimately with
       the sequential API too -- so sort an index here before emitting
       the XML. Insertion sort: part counts are tiny (<= 10000). */
    size_t pc = m->part_count;
    size_t *ord = malloc(pc * sizeof *ord);
    if (!ord && pc) return S3_ERR_OOM;
    for (size_t i = 0; i < pc; i++) ord[i] = i;
    for (size_t i = 1; i < pc; i++) {
        size_t k = ord[i];
        int kn = m->part_numbers[k];
        size_t j = i;
        while (j > 0 && m->part_numbers[ord[j - 1]] > kn) {
            ord[j] = ord[j - 1];
            j--;
        }
        ord[j] = k;
    }

    /* Build CompleteMultipartUpload XML in sorted part order. */
    char *xml = xstrdup("<CompleteMultipartUpload>");
    for (size_t i = 0; i < pc; i++) {
        size_t k = ord[i];
        xml = str_appendf(xml,
            "<Part><PartNumber>%d</PartNumber><ETag>%s</ETag></Part>",
            m->part_numbers[k], m->etags[k]);
    }
    xml = str_appendf(xml, "</CompleteMultipartUpload>");
    free(ord);

    char *u = str_appendf(NULL, "%s?uploadId=%s", m->url, m->upload_id);
    s3_status rc = do_request(m->client, u, M_POST, xml, strlen(xml),
                              NULL, 0, "application/xml", NULL, NULL, NULL, resp);
    free(u);
    free(xml);
    mp_free(m);
    return rc;
}

s3_status s3_multipart_abort(s3_multipart *m) {
    if (!m) return S3_ERR_INVALID_ARG;
    char *u = str_appendf(NULL, "%s?uploadId=%s", m->url, m->upload_id);
    s3_response r = {0};
    s3_status rc = do_request(m->client, u, M_DELETE, NULL, 0, NULL, 0, NULL, NULL,
                              NULL, NULL, &r);
    s3_response_free(&r);
    free(u);
    mp_free(m);
    return rc;
}

/* ================================================================== */
/* ListObjectsV2                                                       */
/* ================================================================== */

void s3_list_result_free(s3_list_result *r) {
    if (!r) return;
    for (size_t i = 0; i < r->prefix_count; i++) free(r->prefixes[i]);
    free(r->prefixes);
    for (size_t i = 0; i < r->object_count; i++) {
        free(r->objects[i].key);
        free(r->objects[i].etag);
    }
    free(r->objects);
    free(r->next_continuation_token);
    memset(r, 0, sizeof *r);
}

s3_status s3_list_ex(s3_client *c, const char *s3_url_prefix,
                     const s3_list_params *params, s3_list_result *out) {
    if (!c || !s3_url_prefix || !out) return S3_ERR_INVALID_ARG;
    memset(out, 0, sizeof *out);

    s3_list_params pz = {0};
    const s3_list_params *p_ = params ? params : &pz;

    s3_url u;
    if (s3_url_parse(s3_url_prefix, &u) != S3_OK) return S3_ERR_INVALID_ARG;
    const char *region = (u.region && u.region[0]) ? u.region : "us-east-1";

    /* The list operation targets the bucket root with query params; honour
       a non-AWS endpoint (path-style) the same way object ops do. */
    char *url;
    if (c->endpoint && c->endpoint[0])
        url = str_appendf(NULL,
            "%s://%s/%s?list-type=2&encoding-type=url",
            c->endpoint_insecure ? "http" : "https",
            c->endpoint, u.bucket);
    else
        url = str_appendf(NULL,
            "https://%s.s3.%s.amazonaws.com/?list-type=2&encoding-type=url",
            u.bucket, region);

    if (p_->delimiter && p_->delimiter[0]) {
        char *e = url_encode(p_->delimiter);
        url = str_appendf(url, "&delimiter=%s", e);
        free(e);
    }
    if (u.key && u.key[0]) {
        char *e = url_encode(u.key);
        url = str_appendf(url, "&prefix=%s", e);
        free(e);
    }
    if (p_->continuation_token && p_->continuation_token[0]) {
        char *e = url_encode(p_->continuation_token);
        url = str_appendf(url, "&continuation-token=%s", e);
        free(e);
    }
    if (p_->start_after && p_->start_after[0]) {
        char *e = url_encode(p_->start_after);
        url = str_appendf(url, "&start-after=%s", e);
        free(e);
    }
    if (p_->max_keys > 0)
        url = str_appendf(url, "&max-keys=%d", p_->max_keys);
    s3_url_free(&u);

    s3_response r = {0};
    s3_status rc = do_request(c, url, M_GET, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, &r);
    free(url);
    if (rc != S3_OK) { s3_response_free(&r); return rc; }
    if (!r.body) { s3_response_free(&r); return S3_ERR_PARSE; }

    const char *body = (const char *)r.body;

    /* CommonPrefixes -> <Prefix> */
    const char *p = body;
    const char *cpend;
    while ((p = strstr(p, "<CommonPrefixes>")) != NULL) {
        const char *blockend = strstr(p, "</CommonPrefixes>");
        char *pref = xml_tag(p, "Prefix", &cpend);
        if (pref) {
            url_decode_inplace(pref);
            out->prefixes = realloc(out->prefixes,
                                    (out->prefix_count + 1) * sizeof(char *));
            out->prefixes[out->prefix_count++] = pref;
        }
        p = blockend ? blockend + 17 : (p + 16);
    }

    /* Contents -> Key/Size/ETag */
    p = body;
    while ((p = strstr(p, "<Contents>")) != NULL) {
        const char *blockend = strstr(p, "</Contents>");
        size_t blen = blockend ? (size_t)(blockend - p) : strlen(p);
        char *block = strndup_(p, blen);
        char *key  = xml_tag(block, "Key", NULL);
        char *size = xml_tag(block, "Size", NULL);
        char *etag = xml_tag(block, "ETag", NULL);
        free(block);
        if (key) {
            url_decode_inplace(key);
            out->objects = realloc(out->objects,
                              (out->object_count + 1) * sizeof(s3_object));
            s3_object *o = &out->objects[out->object_count++];
            o->key = key;
            o->size = size ? strtoull(size, NULL, 10) : 0;
            o->etag = etag ? etag : NULL;
            etag = NULL;
        }
        free(size); free(etag);
        p = blockend ? blockend + 11 : (p + 10);
    }

    char *trunc = xml_tag(body, "IsTruncated", NULL);
    out->is_truncated = trunc && strcmp(trunc, "true") == 0;
    free(trunc);
    out->next_continuation_token =
        xml_tag(body, "NextContinuationToken", NULL);

    s3_response_free(&r);
    return S3_OK;
}

s3_status s3_list(s3_client *c, const char *s3_url_prefix,
                  const char *delimiter, const char *continuation_token,
                  s3_list_result *out) {
    s3_list_params p = {
        .delimiter = delimiter,
        .continuation_token = continuation_token,
    };
    return s3_list_ex(c, s3_url_prefix, &p, out);
}

s3_status s3_list_all(s3_client *c, const char *s3_url_prefix,
                      const char *delimiter,
                      s3_list_page_fn cb, void *userdata) {
    if (!cb) return S3_ERR_INVALID_ARG;
    char *token = NULL;
    for (;;) {
        s3_list_result page;
        s3_status rc = s3_list(c, s3_url_prefix, delimiter, token, &page);
        free(token);
        token = NULL;
        if (rc != S3_OK) return rc;
        bool cont = cb(userdata, &page);
        bool more = page.is_truncated && page.next_continuation_token;
        if (more) token = xstrdup(page.next_continuation_token);
        s3_list_result_free(&page);
        if (!cont || !more) break;
    }
    free(token);
    return S3_OK;
}
