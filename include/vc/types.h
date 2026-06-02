// Fundamental fixed-width types and tiny helpers shared across the toolkit.
// Pure C23, libc only.
#ifndef VC_TYPES_H
#define VC_TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;

// A logical 3D extent / coordinate in voxels, row-major (Z slowest, X fastest).
typedef struct {
    u32 z, y, x;
} vc_coord3;

#endif // VC_TYPES_H
