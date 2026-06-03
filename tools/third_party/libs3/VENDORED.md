# Vendored libs3

Source: https://github.com/SuperOptimizer/libs3
Commit: 22bf07e9a16ec054b7e421098c88aab082ce38b5

Single-file C S3 client (libcurl-backed). Vendored here (like cJSON) so the
exporter builds self-contained on EC2 with no external path dependency.

Includes the thread-safety fixes required for many-threaded S3 access:
curl_global_init mutex, CURLOPT_NOSIGNAL, low-speed stall watchdog, and the
s3_get_to_file streaming API. To update: re-copy from upstream and bump the
commit hash above.
