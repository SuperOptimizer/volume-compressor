/*
 * Internal white-box test surface.
 *
 * The scraper / parsing helpers in libs3.c are `static` in normal builds.
 * White-box unit tests (test_parsers, test_creds) need to call them
 * directly without #including the whole .c (which would create a second
 * instrumented instance and corrupt coverage numbers).
 *
 * libs3.c defines LIBS3_INTERNAL as `static` normally, or empty when
 * built with -DLIBS3_TESTING, giving these helpers external linkage.
 * Tests then link the instrumented library and declare them via this
 * header. Production builds are unaffected (default = static).
 */
#ifndef LIBS3_INTERNAL_H
#define LIBS3_INTERNAL_H

#include "libs3.h"
#include <time.h>

#ifdef LIBS3_TESTING

char  *json_string_field(const char *json, const char *key);
char  *xml_tag(const char *from, const char *tag, const char **end);
void   url_decode_inplace(char *s);
char  *url_encode(const char *s);
time_t parse_iso8601(const char *s);
void   find_sso_profiles(char ***out, size_t *n);

/* IMDS test hooks. The endpoint is overridable via $LIBS3_IMDS_BASE so a
 * local mock can stand in; these drive the fetch/cache paths directly. */
void libs3_test_reset_imds_cache(void);
bool libs3_test_fetch_imds(s3_credentials *out, time_t *expiry);
bool libs3_test_cached_imds(s3_credentials *out);

#endif /* LIBS3_TESTING */

#endif /* LIBS3_INTERNAL_H */
