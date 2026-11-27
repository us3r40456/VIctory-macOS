#!/bin/sh
# Builds the Route A.5 Wine unixlib bridge (victory_metal.dll + victory_metal.so)
# for zero-copy Apple Silicon unified memory, global MTLResidencySet tracking,
# in-process DXIL -> MetalLib conversion via libmetalirconverter.dylib,
# native MetalFX Temporal Scaling & Frame Interpolation, and CAMetalLayer
# presentation.
#
# Usage:
#   tools/macos/victory-metal-unixlib.sh OUTPUT_DIR [WINE_ROOT]
#
# Produces:
#   OUTPUT_DIR/x86_64-windows/victory_metal.dll  (PE builtin DLL)
#   OUTPUT_DIR/x86_64-unix/victory_metal.so      (x86_64 Mach-O Metal unixlib)
#
# Requires no Wine or ntdll rebuild: prepend OUTPUT_DIR to WINEDLLPATH.
# Kept as a shell script so the repository asset policy stays clean.
set -e

out_dir="$1"
wine_root="${2:-}"
[ -n "$out_dir" ] || { echo "usage: $0 OUTPUT_DIR [WINE_ROOT]" >&2; exit 2; }

if [ -z "$wine_root" ]; then
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    repo_root="$(cd "$script_dir/../.." && pwd)"
    if [ -d "$repo_root/target/wine-cx/include/wine" ]; then
        wine_root="$repo_root/target/wine-cx"
    elif [ -d "$HOME/Library/Caches/victory-macos/target/wine-cx/include/wine" ]; then
        wine_root="$HOME/Library/Caches/victory-macos/target/wine-cx"
    fi
fi

[ -f "$wine_root/include/wine/unixlib.h" ] || {
    echo "error: Wine headers not found under '$wine_root/include/wine'" >&2
    exit 1
}

winegcc="$wine_root/bin/winegcc"
[ -x "$winegcc" ] || { echo "error: winegcc not found at $winegcc" >&2; exit 1; }

mkdir -p "$out_dir/x86_64-windows" "$out_dir/x86_64-unix"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/victory-metal-unixlib.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

# ---------------------------------------------------------------------------
# 1. PE Builtin DLL (x86_64-windows/victory_metal.dll)
# ---------------------------------------------------------------------------
cat >"$tmp/victory_metal.spec" <<'SPEC'
@ stdcall victory_metal_available()
@ stdcall victory_metal_call(long ptr)
SPEC

cat >"$tmp/victory_metal_pe.c" <<'PE_C'
#define WIN32_LEAN_AND_MEAN
#include <stdint.h>
#include <windows.h>
#include <winternl.h>
#include <wine/unixlib.h>

static volatile LONG g_unix_ready = 0;

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);
        if (__wine_init_unix_call() == 0) {
            InterlockedExchange(&g_unix_ready, 1);
        }
    }
    return TRUE;
}

int32_t WINAPI victory_metal_available(void) {
    return InterlockedCompareExchange(&g_unix_ready, 0, 0) ? 1 : 0;
}

int32_t WINAPI victory_metal_call(uint32_t opcode, void *args) {
    if (!InterlockedCompareExchange(&g_unix_ready, 0, 0)) {
        return (int32_t)0xC0000001u; /* STATUS_UNSUCCESSFUL */
    }
    return (int32_t)WINE_UNIX_CALL(opcode, args);
}
PE_C

PATH="/opt/homebrew/bin:$wine_root/bin:$PATH" \
"$winegcc" -b x86_64-w64-mingw32 -shared -mno-cygwin -O2 \
    -I"$wine_root/include" \
    -L"$wine_root/lib/wine/x86_64-windows" \
    "$tmp/victory_metal.spec" \
    "$tmp/victory_metal_pe.c" \
    -lntdll \
    -o "$out_dir/x86_64-windows/victory_metal.dll"

"$wine_root/bin/winebuild" --builtin "$out_dir/x86_64-windows/victory_metal.dll"

# ---------------------------------------------------------------------------
# 2. Mach-O Unixlib (x86_64-unix/victory_metal.so)
# ---------------------------------------------------------------------------
cat >"$tmp/victory_metal_unix.mm" <<'UNIX_MM'
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import <dispatch/dispatch.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

typedef int32_t NTSTATUS;
typedef NTSTATUS (*unixlib_entry_t)(void *args);

static constexpr NTSTATUS VM_STATUS_SUCCESS = 0;
static constexpr NTSTATUS VM_STATUS_INVALID_PARAM = (NTSTATUS)0xC000000Du;
static constexpr NTSTATUS VM_STATUS_UNSUCCESSFUL = (NTSTATUS)0xC0000001u;
static constexpr uint32_t VM_ABI_VERSION = 3;

// Must match crates/victory-gpu/src/metal/mod.rs
struct vm_ping_args {
    uint32_t abi_version;
    uint32_t flags;
    uint64_t recommended_vram;
    uint64_t max_buffer_len;
    char device_name[128];
};

struct vm_register_buffer_args {
    uint64_t cpu_addr;
    uint64_t length;
    uint64_t backing_key;
    uint32_t flags;
    uint32_t reserved;
    uint64_t handle;
    uint64_t gpu_address;
    uint32_t reused;
    int32_t status;
};

struct vm_unregister_buffer_args {
    uint64_t handle;
    uint32_t flags;
    uint32_t remaining_refs;
};

struct vm_flush_residency_args {
    uint32_t pending_before;
    uint32_t total_allocations;
};

struct vm_bench_args {
    uint64_t handle;
    uint64_t offset;
    uint64_t length;
    uint32_t mode;
    uint32_t seed;
    uint64_t checksum_out;
    uint64_t gpu_ns;
    uint64_t wall_ns;
};

struct vm_convert_dxil_args {
    uint64_t dxil_ptr;
    uint64_t dxil_len;
    uint64_t entry_ptr;
    uint64_t out_ptr;
    uint64_t out_cap;
    uint64_t out_len;
    uint32_t stage;
    int32_t status;
};

struct vm_stats_args {
    uint64_t active_buffers;
    uint64_t active_bytes;
    uint64_t total_registered;
    uint64_t total_reused;
    uint64_t residency_commits;
    uint64_t dxil_converted;
};

struct vm_create_texture_args {
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format; // MTLPixelFormat numeric value
    uint32_t usage;        // MTLTextureUsage mask (0 = default RW+RT)
    uint32_t storage_mode; // 0 = Private, 1 = Shared
    int32_t status;
    uint64_t handle;
};

struct vm_destroy_handle_args {
    uint64_t handle;
};

struct vm_texture_io_args {
    uint64_t texture_handle;
    uint64_t buffer_handle; // optional registered MTLBuffer (0 => use cpu_ptr)
    uint64_t cpu_ptr;       // raw CPU address when buffer_handle == 0
    uint64_t offset;        // byte offset inside buffer_handle
    uint32_t bytes_per_row;
    uint32_t width;
    uint32_t height;
    int32_t status;
};

struct vm_scaler_create_args {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t color_format;
    uint32_t depth_format;
    uint32_t motion_format;
    uint32_t output_format;
    uint32_t reactive_format; // 0 = disabled
    uint32_t flags;           // bit 0: auto_exposure, bit 1: sync_init
    uint64_t handle;
    int32_t status;
    uint32_t reserved;
};

struct vm_scaler_encode_args {
    uint64_t scaler_handle;
    uint64_t color_tex;
    uint64_t depth_tex;
    uint64_t motion_tex;
    uint64_t reactive_tex;
    uint64_t output_tex;
    uint32_t input_content_width;
    uint32_t input_content_height;
    float jitter_x;
    float jitter_y;
    float mv_scale_x;
    float mv_scale_y;
    float pre_exposure;
    uint32_t flags; // bit 0: reset, bit 1: depth_reversed
    uint64_t gpu_ns;
    uint64_t wall_ns;
    int32_t status;
    uint32_t reserved;
};

struct vm_interp_create_args {
    uint32_t input_width;  // depth/motion width (0 => output_width/2 for flow)
    uint32_t input_height; // depth/motion height
    uint32_t output_width; // color/output width
    uint32_t output_height;
    uint32_t color_format;
    uint32_t output_format;
    uint32_t depth_format;
    uint32_t motion_format;
    uint32_t ui_format; // 0 = no UI texture
    uint32_t flags;
    uint64_t handle;
    int32_t status;
    uint32_t reserved;
};

struct vm_interp_encode_args {
    uint64_t interp_handle;
    uint64_t color_tex;
    uint64_t prev_color_tex; // 0 = use interpolator's internal ping-pong prev
    uint64_t depth_tex;      // 0 = synthetic flat depth
    uint64_t motion_tex;     // 0 = built-in Metal compute optical flow
    uint64_t ui_tex;         // 0 = no UI plane (any size; auto-resampled)
    uint64_t output_tex;
    float jitter_x;
    float jitter_y;
    float mv_scale_x;
    float mv_scale_y;
    float delta_time;
    float near_plane;
    float far_plane;
    float fov_degrees;
    uint32_t flags; // bit 0: reset, bit 1: depth_reversed, bit 2: ui_composited, bit 3: skip_prev_update
    uint32_t interpolated; // out: 1 if an interpolated frame was generated
    uint64_t gpu_ns;
    uint64_t wall_ns;
    int32_t status;
    uint32_t reserved;
};

struct vm_surface_attach_args {
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format; // 0 => BGRA8Unorm (80)
    uint32_t flags;        // bit 0: vsync/displaySyncEnabled
    uint64_t handle;
    int32_t status;
    uint32_t reserved;
};

struct vm_surface_present_args {
    uint64_t surface_handle;
    uint64_t color_tex;
    uint64_t ui_tex; // optional Plane 1 overlay composited on present
    uint64_t after_min_duration_ns;
    uint32_t flags; // bit 0: wait_completed
    int32_t status;
    uint64_t gpu_ns;
    uint64_t wall_ns;
};

// Step 2B-3: MTLBinaryArchive, native compute/render PSOs, and batched
// MTLCommandBuffer command stream submission.
struct vm_archive_open_args {
    uint64_t path_ptr; // UTF-8 POSIX path (0 => in-memory archive)
    uint64_t handle;
    uint32_t loaded_existing;
    int32_t status;
};

struct vm_archive_save_args {
    uint64_t handle;
    uint64_t path_ptr;
    int32_t status;
    uint32_t reserved;
};

struct vm_pipeline_compute_args {
    uint64_t bytes_ptr;
    uint64_t bytes_len;
    uint64_t entry_ptr;      // UTF-8 function name (0 => "main")
    uint64_t archive_handle; // optional MTLBinaryArchive (0 => none)
    uint32_t source_kind;    // 0 = DXIL, 1 = MetalLib (MTLB), 2 = MSL source
    int32_t status;
    uint64_t handle;
    uint32_t max_threads_per_tg;
    uint32_t simd_width;
};

struct vm_pipeline_render_args {
    uint64_t vs_ptr;
    uint64_t vs_len;
    uint64_t vs_entry_ptr;
    uint64_t fs_ptr;
    uint64_t fs_len;
    uint64_t fs_entry_ptr;
    uint64_t archive_handle;
    uint32_t source_kind;  // 0 = DXIL, 1 = MetalLib, 2 = MSL source
    uint32_t color_format; // MTLPixelFormat (0 => RGBA8Unorm)
    uint32_t depth_format; // MTLPixelFormat (0 => none)
    uint32_t flags;        // bit 0: alpha_blend
    uint64_t handle;
    int32_t status;
    uint32_t reserved;
};

struct vm_submit_commands_args {
    uint64_t stream_ptr; // pointer to packed uint32_t command words
    uint32_t word_count;
    uint32_t flags; // bit 0: wait_completed
    uint64_t fence_value;
    uint64_t encode_ns;
    uint64_t gpu_ns;
    uint64_t wall_ns;
    uint32_t draws_encoded;
    uint32_t dispatches_encoded;
    int32_t status;
    uint32_t reserved;
};

struct vm_wait_fence_args {
    uint64_t fence_value;
    uint64_t timeout_ns;
    uint32_t signaled;
    int32_t status;
};

// Packed uint32_t command stream opcodes for OP_SUBMIT_COMMANDS.
static constexpr uint32_t CMD_COMPUTE_BEGIN = 1;
static constexpr uint32_t CMD_COMPUTE_END = 2;
static constexpr uint32_t CMD_SET_COMPUTE_PSO = 3;
static constexpr uint32_t CMD_SET_COMPUTE_BUFFER = 4;
static constexpr uint32_t CMD_SET_COMPUTE_BYTES = 5;
static constexpr uint32_t CMD_SET_COMPUTE_TEXTURE = 6;
static constexpr uint32_t CMD_DISPATCH_TG = 7;
static constexpr uint32_t CMD_DISPATCH_INDIRECT = 8;
static constexpr uint32_t CMD_BLIT_COPY_BUFFER = 9;
static constexpr uint32_t CMD_BLIT_FILL_BUFFER = 10;
static constexpr uint32_t CMD_RENDER_BEGIN = 11;
static constexpr uint32_t CMD_RENDER_END = 12;
static constexpr uint32_t CMD_SET_RENDER_PSO = 13;
static constexpr uint32_t CMD_SET_VIEWPORT = 14;
static constexpr uint32_t CMD_SET_VERTEX_BUFFER = 15;
static constexpr uint32_t CMD_SET_VERTEX_BYTES = 16;
static constexpr uint32_t CMD_SET_FRAGMENT_BUFFER = 17;
static constexpr uint32_t CMD_SET_FRAGMENT_BYTES = 18;
static constexpr uint32_t CMD_SET_FRAGMENT_TEXTURE = 19;
static constexpr uint32_t CMD_DRAW_PRIMITIVES = 20;
static constexpr uint32_t CMD_DRAW_INDEXED = 21;
static constexpr uint32_t CMD_MEMORY_BARRIER = 22;

// Transparent-to-input CAMetalLayer overlay view attached to Wine's NSWindow
// so keyboard, mouse, and controller focus stay 100% untouched.
@interface VictoryMetalOverlayView : NSView
@property (nonatomic, strong) CAMetalLayer *metalLayer;
@end

@implementation VictoryMetalOverlayView
- (BOOL)wantsUpdateLayer {
    return YES;
}
- (CALayer *)makeBackingLayer {
    CAMetalLayer *layer = [CAMetalLayer layer];
    self.metalLayer = layer;
    return layer;
}
- (NSView *)hitTest:(NSPoint)point {
    (void)point;
    return nil;
}
@end

// Dynamic bindings to Apple's Metal Shader Converter (libmetalirconverter.dylib)
namespace ir {
struct Compiler;
struct Object;
struct MetalLibBinary;
struct Error;

using FnCompilerCreate = Compiler *(*)();
using FnCompilerDestroy = void (*)(Compiler *);
using FnCompilerSetEntry = void (*)(Compiler *, const char *);
using FnCompilerSetTarget = void (*)(Compiler *, uint32_t, const char *);
using FnCompilerCompile = Object *(*)(Compiler *, const char *, Object *, Error **);
using FnObjectFromDXIL = Object *(*)(const uint8_t *, size_t, uint32_t);
using FnObjectDestroy = void (*)(Object *);
using FnObjectGetStage = uint32_t (*)(const Object *);
using FnObjectGetMetalLib = bool (*)(const Object *, uint32_t, MetalLibBinary *);
using FnMetalLibCreate = MetalLibBinary *(*)();
using FnMetalLibDestroy = void (*)(MetalLibBinary *);
using FnMetalLibSize = size_t (*)(const MetalLibBinary *);
using FnMetalLibBytes = size_t (*)(const MetalLibBinary *, uint8_t *);
using FnErrorDestroy = void (*)(Error *);

struct Api {
    void *dl = nullptr;
    FnCompilerCreate compiler_create = nullptr;
    FnCompilerDestroy compiler_destroy = nullptr;
    FnCompilerSetEntry compiler_set_entry = nullptr;
    FnCompilerSetTarget compiler_set_target = nullptr;
    FnCompilerCompile compiler_compile = nullptr;
    FnObjectFromDXIL object_from_dxil = nullptr;
    FnObjectDestroy object_destroy = nullptr;
    FnObjectGetStage object_get_stage = nullptr;
    FnObjectGetMetalLib object_get_metallib = nullptr;
    FnMetalLibCreate metallib_create = nullptr;
    FnMetalLibDestroy metallib_destroy = nullptr;
    FnMetalLibSize metallib_size = nullptr;
    FnMetalLibBytes metallib_bytes = nullptr;
    FnErrorDestroy error_destroy = nullptr;

    bool loaded() const {
        return dl && compiler_create && compiler_destroy && compiler_compile &&
               object_from_dxil && object_destroy && object_get_stage &&
               object_get_metallib && metallib_create && metallib_destroy &&
               metallib_size && metallib_bytes;
    }
};

static Api &api() {
    static Api a;
    static std::once_flag once;
    std::call_once(once, []() {
        std::vector<std::string> candidates;
        if (const char *env = std::getenv("VICTORY_METAL_IR_LIB")) {
            if (*env) candidates.emplace_back(env);
        }
        if (const char *dxc = std::getenv("DXC_LIB")) {
            std::string s(dxc);
            auto pos = s.rfind('/');
            if (pos != std::string::npos) {
                candidates.emplace_back(s.substr(0, pos + 1) + "libmetalirconverter.dylib");
            }
        }
        candidates.emplace_back("/usr/local/lib/libmetalirconverter.dylib");
        candidates.emplace_back("libmetalirconverter.dylib");

        for (const auto &path : candidates) {
            void *h = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h) continue;
            a.dl = h;
            a.compiler_create = (FnCompilerCreate)dlsym(h, "IRCompilerCreate");
            a.compiler_destroy = (FnCompilerDestroy)dlsym(h, "IRCompilerDestroy");
            a.compiler_set_entry = (FnCompilerSetEntry)dlsym(h, "IRCompilerSetEntryPointName");
            a.compiler_set_target = (FnCompilerSetTarget)dlsym(h, "IRCompilerSetMinimumDeploymentTarget");
            a.compiler_compile = (FnCompilerCompile)dlsym(h, "IRCompilerAllocCompileAndLink");
            a.object_from_dxil = (FnObjectFromDXIL)dlsym(h, "IRObjectCreateFromDXIL");
            a.object_destroy = (FnObjectDestroy)dlsym(h, "IRObjectDestroy");
            a.object_get_stage = (FnObjectGetStage)dlsym(h, "IRObjectGetMetalIRShaderStage");
            a.object_get_metallib = (FnObjectGetMetalLib)dlsym(h, "IRObjectGetMetalLibBinary");
            a.metallib_create = (FnMetalLibCreate)dlsym(h, "IRMetalLibBinaryCreate");
            a.metallib_destroy = (FnMetalLibDestroy)dlsym(h, "IRMetalLibBinaryDestroy");
            a.metallib_size = (FnMetalLibSize)dlsym(h, "IRMetalLibGetBytecodeSize");
            a.metallib_bytes = (FnMetalLibBytes)dlsym(h, "IRMetalLibGetBytecode");
            a.error_destroy = (FnErrorDestroy)dlsym(h, "IRErrorDestroy");
            if (a.loaded()) break;
            dlclose(h);
            a = Api{};
        }
    });
    return a;
}
} // namespace ir

struct BufferEntry {
    uint64_t handle = 0;
    uint64_t cpu_addr = 0;
    uint64_t length = 0;
    uint64_t backing_key = 0;
    uint64_t gpu_address = 0;
    uint32_t refcount = 0;
    id<MTLBuffer> buffer = nil;
};

struct TextureEntry {
    uint64_t handle = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    MTLPixelFormat format = MTLPixelFormatInvalid;
    id<MTLTexture> texture = nil;
};

struct ScalerEntry {
    uint64_t handle = 0;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    id<MTLFXTemporalScaler> scaler = nil;
};

struct InterpEntry {
    uint64_t handle = 0;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    MTLPixelFormat color_format = MTLPixelFormatInvalid;
    MTLPixelFormat ui_format = MTLPixelFormatInvalid;
    id interpolator = nil; // id<MTLFXFrameInterpolator> on macOS 26+
    id<MTLTexture> prev_color = nil;
    id<MTLTexture> flow_motion = nil;
    id<MTLTexture> const_depth = nil;
    id<MTLTexture> resampled_ui = nil;
    bool has_prev = false;
};

struct SurfaceEntry {
    uint64_t handle = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    VictoryMetalOverlayView *view = nil;
    CAMetalLayer *layer = nil;
};

struct ArchiveEntry {
    uint64_t handle = 0;
    id<MTLBinaryArchive> archive = nil;
    std::string path;
};

struct ComputePsoEntry {
    uint64_t handle = 0;
    id<MTLComputePipelineState> pso = nil;
};

struct RenderPsoEntry {
    uint64_t handle = 0;
    id<MTLRenderPipelineState> pso = nil;
};

static uint64_t now_ns() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (mach_absolute_time() * tb.numer) / tb.denom;
}

static bool run_on_main_bounded(void (^block)(void), int64_t timeout_ms) {
    if ([NSThread isMainThread]) {
        block();
        return true;
    }
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            block();
        }
        dispatch_semaphore_signal(sem);
    });
    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, timeout_ms * NSEC_PER_MSEC);
    return dispatch_semaphore_wait(sem, deadline) == 0;
}

static const char *BUILTIN_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void bench_sum(
    device const uint *src [[buffer(0)]],
    device ulong *partials [[buffer(1)]],
    constant uint &word_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgsize [[threads_per_threadgroup]]
) {
    threadgroup ulong shared_sum[256];
    ulong acc = 0;
    for (uint i = gid; i < word_count; i += gsize) {
        acc += (ulong)src[i];
    }
    shared_sum[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partials[tgid] = shared_sum[0];
    }
}

kernel void bench_write(
    device uint *dst [[buffer(0)]],
    constant uint &word_count [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]]
) {
    for (uint i = gid; i < word_count; i += gsize) {
        dst[i] = seed ^ i;
    }
}

// Fast 4-step logarithmic search (steps 8, 4, 2, 1) with 5-point cross patch
// for present-time Frame Interpolation when engine motion vectors are not
// supplied. Matches MetalFX convention: object moving (+dx,+dy) in curr has
// backward motion vector (-dx,-dy) pointing to prevColorTexture.
static inline float luma_at(texture2d<float, access::sample> tex, sampler smp, float2 uv) {
    float3 c = tex.sample(smp, uv).rgb;
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

static inline float cross_sad(
    texture2d<float, access::sample> prev_tex,
    sampler smp,
    float2 uv,
    float2 inv_m,
    float c0, float c1, float c2, float c3, float c4
) {
    float e = abs(c0 - luma_at(prev_tex, smp, uv));
    e += abs(c1 - luma_at(prev_tex, smp, uv + float2(inv_m.x, 0.0f)));
    e += abs(c2 - luma_at(prev_tex, smp, uv - float2(inv_m.x, 0.0f)));
    e += abs(c3 - luma_at(prev_tex, smp, uv + float2(0.0f, inv_m.y)));
    e += abs(c4 - luma_at(prev_tex, smp, uv - float2(0.0f, inv_m.y)));
    return e;
}

kernel void optical_flow_mv(
    texture2d<float, access::sample> curr_tex [[texture(0)]],
    texture2d<float, access::sample> prev_tex [[texture(1)]],
    texture2d<float, access::write> motion_out [[texture(2)]],
    texture2d<float, access::write> depth_out [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint mw = motion_out.get_width();
    uint mh = motion_out.get_height();
    if (gid.x >= mw || gid.y >= mh) return;

    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 inv_m = float2(1.0f / float(mw), 1.0f / float(mh));
    float2 uv = (float2(gid) + 0.5f) * inv_m;

    float c0 = luma_at(curr_tex, smp, uv);
    float c1 = luma_at(curr_tex, smp, uv + float2(inv_m.x, 0.0f));
    float c2 = luma_at(curr_tex, smp, uv - float2(inv_m.x, 0.0f));
    float c3 = luma_at(curr_tex, smp, uv + float2(0.0f, inv_m.y));
    float c4 = luma_at(curr_tex, smp, uv - float2(0.0f, inv_m.y));

    float2 best_off = float2(0.0f);
    float best_err = cross_sad(prev_tex, smp, uv, inv_m, c0, c1, c2, c3, c4) * 0.94f;

    // Early out on static pixels (zero motion).
    if (best_err > 0.015f) {
        const float steps[4] = { 8.0f, 4.0f, 2.0f, 1.0f };
        for (int si = 0; si < 4; ++si) {
            float step = steps[si];
            float2 center = best_off;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    float2 cand = center + float2(float(dx) * step, float(dy) * step);
                    float err = cross_sad(prev_tex, smp, uv + cand * inv_m, inv_m, c0, c1, c2, c3, c4);
                    if (err < best_err) {
                        best_err = err;
                        best_off = cand;
                    }
                }
            }
        }
    }

    float2 scale = float2(float(curr_tex.get_width()) * inv_m.x, float(curr_tex.get_height()) * inv_m.y);
    float2 mv_px = best_off * scale;
    motion_out.write(float4(mv_px.x, mv_px.y, 0.0f, 0.0f), gid);
    depth_out.write(float4(0.5f, 0.0f, 0.0f, 0.0f), gid);
}

// Resamples Plane 1 UI overlay (e.g. 3840x2160) cleanly to output dimensions.
kernel void resample_ui(
    texture2d<float, access::sample> src_ui [[texture(0)]],
    texture2d<float, access::write> dst_ui [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dw = dst_ui.get_width();
    uint dh = dst_ui.get_height();
    if (gid.x >= dw || gid.y >= dh) return;
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(float(dw), float(dh));
    dst_ui.write(src_ui.sample(smp, uv), gid);
}

// Composites Plane 0 scene + optional Plane 1 UI directly into a CAMetalLayer
// drawable texture (handles scaling and alpha blend in a single pass).
kernel void composite_present(
    texture2d<float, access::sample> color_tex [[texture(0)]],
    texture2d<float, access::sample> ui_tex [[texture(1)]],
    texture2d<float, access::write> dst_tex [[texture(2)]],
    constant uint &has_ui [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dw = dst_tex.get_width();
    uint dh = dst_tex.get_height();
    if (gid.x >= dw || gid.y >= dh) return;
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) / float2(float(dw), float(dh));
    float4 c = color_tex.sample(smp, uv);
    if (has_ui != 0) {
        float4 o = ui_tex.sample(smp, uv);
        c = float4(mix(c.rgb, o.rgb, o.a), 1.0f);
    } else {
        c.a = 1.0f;
    }
    dst_tex.write(c, gid);
}
)MSL";

struct MetalState {
    std::mutex mu;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    id<MTLComputePipelineState> bench_sum_pso = nil;
    id<MTLComputePipelineState> bench_write_pso = nil;
    id<MTLComputePipelineState> optical_flow_pso = nil;
    id<MTLComputePipelineState> resample_ui_pso = nil;
    id<MTLComputePipelineState> composite_present_pso = nil;
    id<MTLBuffer> bench_partials = nil;
    id<MTLBuffer> staging_buf = nil;
    size_t staging_cap = 0;

    uint64_t next_handle = 1;
    uint32_t pending_residency = 0;
    bool has_residency = false;
    bool has_metalfx = false;
    bool has_frame_interp = false;
    bool has_metal4_fx = false;
    uint32_t cached_flags = 0;
    uint64_t cached_vram = 0;
    uint64_t cached_max_buf = 0;
    char cached_name[128] = {0};

    std::unordered_map<uint64_t, BufferEntry> by_handle;
    std::unordered_map<uint64_t, uint64_t> backing_to_handle;
    std::unordered_map<uint64_t, TextureEntry> textures;
    std::unordered_map<uint64_t, ScalerEntry> scalers;
    std::unordered_map<uint64_t, InterpEntry> interps;
    std::unordered_map<uint64_t, SurfaceEntry> surfaces;
    std::unordered_map<uint64_t, ArchiveEntry> archives;
    std::unordered_map<uint64_t, ComputePsoEntry> compute_psos;
    std::unordered_map<uint64_t, RenderPsoEntry> render_psos;
    id<MTLSharedEvent> timeline_event = nil;
    uint64_t timeline_value = 0;

    uint64_t active_bytes = 0;
    uint64_t total_registered = 0;
    uint64_t total_reused = 0;
    uint64_t residency_commits = 0;
    std::atomic<uint64_t> dxil_converted{0};

    bool init_locked() {
        if (device) return true;
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return false;
            queue = [device newCommandQueue];
            if (!queue) return false;
            timeline_event = [device newSharedEvent];

            if (@available(macOS 15.0, *)) {
                MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
                desc.label = @"victory.global.residency";
                desc.initialCapacity = 256;
                NSError *err = nil;
                residency = [device newResidencySetWithDescriptor:desc error:&err];
                if (residency) {
                    [residency commit];
                    [queue addResidencySet:residency];
                    has_residency = true;
                }
            }
            if (@available(macOS 13.0, *)) {
                has_metalfx = [MTLFXTemporalScalerDescriptor supportsDevice:device];
            }
            if (@available(macOS 26.0, *)) {
                has_frame_interp = [MTLFXFrameInterpolatorDescriptor supportsDevice:device];
                has_metal4_fx = [MTLFXFrameInterpolatorDescriptor supportsMetal4FX:device];
            }
            uint32_t f = 0;
            if (device.hasUnifiedMemory) f |= 1u << 0;
            if (has_residency) f |= 1u << 1;
            if (has_metalfx) f |= 1u << 2;
            if (ir::api().loaded()) f |= 1u << 3;
            if (has_frame_interp) f |= 1u << 4;
            if (has_metal4_fx) f |= 1u << 5;
            cached_flags = f;
            cached_vram = (uint64_t)device.recommendedMaxWorkingSetSize;
            cached_max_buf = (uint64_t)device.maxBufferLength;
            std::memset(cached_name, 0, sizeof(cached_name));
            if (const char *name = device.name.UTF8String) {
                std::strncpy(cached_name, name, sizeof(cached_name) - 1);
            }
        }
        return true;
    }

    bool init_kernels_locked() {
        if (bench_sum_pso && bench_write_pso && optical_flow_pso &&
            resample_ui_pso && composite_present_pso && bench_partials) {
            return true;
        }
        @autoreleasepool {
            NSError *err = nil;
            NSString *src = [NSString stringWithUTF8String:BUILTIN_MSL];
            id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
            if (!lib) return false;
            auto make_pso = [&](NSString *fn) -> id<MTLComputePipelineState> {
                id<MTLFunction> f = [lib newFunctionWithName:fn];
                if (!f) return nil;
                NSError *pe = nil;
                return [device newComputePipelineStateWithFunction:f error:&pe];
            };
            bench_sum_pso = make_pso(@"bench_sum");
            bench_write_pso = make_pso(@"bench_write");
            optical_flow_pso = make_pso(@"optical_flow_mv");
            resample_ui_pso = make_pso(@"resample_ui");
            composite_present_pso = make_pso(@"composite_present");
            bench_partials = [device newBufferWithLength:256 * sizeof(uint64_t)
                                                 options:MTLResourceStorageModeShared];
            if (bench_partials && has_residency) {
                if (@available(macOS 15.0, *)) {
                    [residency addAllocation:bench_partials];
                    [residency commit];
                    residency_commits++;
                }
            }
        }
        return bench_sum_pso && bench_write_pso && optical_flow_pso &&
               resample_ui_pso && composite_present_pso && bench_partials;
    }

    id<MTLBuffer> ensure_staging_locked(size_t need) {
        if (staging_buf && staging_cap >= need) return staging_buf;
        size_t cap = std::max<size_t>(need, 16u << 20);
        cap = (cap + 0xFFFFu) & ~size_t(0xFFFFu);
        id<MTLBuffer> next = [device newBufferWithLength:cap options:MTLResourceStorageModeShared];
        if (!next) return nil;
        staging_buf = next;
        staging_cap = cap;
        return staging_buf;
    }

    void commit_residency_locked() {
        if (!has_residency || pending_residency == 0) return;
        if (@available(macOS 15.0, *)) {
            [residency commit];
            pending_residency = 0;
            residency_commits++;
        }
    }
};

static MetalState &state() {
    static MetalState s;
    return s;
}

static NTSTATUS op_ping(void *raw) {
    auto *a = static_cast<vm_ping_args *>(raw);
    if (!a) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    if (!s.device) {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.init_locked()) return VM_STATUS_UNSUCCESSFUL;
    }
    a->abi_version = VM_ABI_VERSION;
    a->flags = s.cached_flags;
    a->recommended_vram = s.cached_vram;
    a->max_buffer_len = s.cached_max_buf;
    std::memcpy(a->device_name, s.cached_name, sizeof(a->device_name));
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_register_buffer(void *raw) {
    auto *a = static_cast<vm_register_buffer_args *>(raw);
    if (!a || a->cpu_addr == 0 || a->length == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }

    const uint64_t key = a->backing_key ? a->backing_key : a->cpu_addr;
    auto it = s.backing_to_handle.find(key);
    if (it != s.backing_to_handle.end()) {
        auto &entry = s.by_handle[it->second];
        entry.refcount++;
        s.total_reused++;
        a->handle = entry.handle;
        a->gpu_address = entry.gpu_address;
        a->reused = 1;
        a->status = 0;
        return VM_STATUS_SUCCESS;
    }

    @autoreleasepool {
        id<MTLBuffer> buf = [s.device newBufferWithBytesNoCopy:(void *)(uintptr_t)a->cpu_addr
                                                        length:(NSUInteger)a->length
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!buf) {
            a->handle = 0;
            a->gpu_address = 0;
            a->reused = 0;
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        if (s.has_residency) {
            if (@available(macOS 15.0, *)) {
                [s.residency addAllocation:buf];
                s.pending_residency++;
                if ((a->flags & 1u) == 0) {
                    s.commit_residency_locked();
                }
            }
        }
        uint64_t h = s.next_handle++;
        BufferEntry entry;
        entry.handle = h;
        entry.cpu_addr = a->cpu_addr;
        entry.length = a->length;
        entry.backing_key = key;
        entry.gpu_address = (uint64_t)buf.gpuAddress;
        entry.refcount = 1;
        entry.buffer = buf;

        s.by_handle.emplace(h, entry);
        s.backing_to_handle.emplace(key, h);
        s.active_bytes += a->length;
        s.total_registered++;

        a->handle = h;
        a->gpu_address = entry.gpu_address;
        a->reused = 0;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_unregister_buffer(void *raw) {
    auto *a = static_cast<vm_unregister_buffer_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.by_handle.find(a->handle);
    if (it == s.by_handle.end()) {
        a->remaining_refs = 0;
        return VM_STATUS_SUCCESS;
    }
    if (it->second.refcount > 1) {
        it->second.refcount--;
        a->remaining_refs = it->second.refcount;
        return VM_STATUS_SUCCESS;
    }
    @autoreleasepool {
        if (s.has_residency && it->second.buffer) {
            if (@available(macOS 15.0, *)) {
                [s.residency removeAllocation:it->second.buffer];
                s.pending_residency++;
                if ((a->flags & 1u) == 0) {
                    s.commit_residency_locked();
                }
            }
        }
        s.active_bytes -= it->second.length;
        s.backing_to_handle.erase(it->second.backing_key);
        s.by_handle.erase(it);
    }
    a->remaining_refs = 0;
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_flush_residency(void *raw) {
    auto *a = static_cast<vm_flush_residency_args *>(raw);
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (a) a->pending_before = s.pending_residency;
    s.commit_residency_locked();
    if (a) a->total_allocations = (uint32_t)s.by_handle.size();
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_bench_zero_copy(void *raw) {
    auto *a = static_cast<vm_bench_args *>(raw);
    if (!a || a->handle == 0 || a->length == 0 || (a->offset & 3) || (a->length & 3)) {
        return VM_STATUS_INVALID_PARAM;
    }
    auto &s = state();
    id<MTLBuffer> buf = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> partials = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.init_locked() || !s.init_kernels_locked()) return VM_STATUS_UNSUCCESSFUL;
        s.commit_residency_locked();
        auto it = s.by_handle.find(a->handle);
        if (it == s.by_handle.end() || a->offset + a->length > it->second.length) {
            return VM_STATUS_INVALID_PARAM;
        }
        buf = it->second.buffer;
        queue = s.queue;
        pso = (a->mode == 0) ? s.bench_sum_pso : s.bench_write_pso;
        partials = s.bench_partials;
    }

    const uint32_t words = (uint32_t)(a->length / 4);
    const uint32_t groups = 256;
    const uint32_t threads_per_group = 256;

    @autoreleasepool {
        uint64_t t0 = now_ns();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        if (a->mode == 0) {
            [enc setBuffer:buf offset:(NSUInteger)a->offset atIndex:0];
            [enc setBuffer:partials offset:0 atIndex:1];
            [enc setBytes:&words length:sizeof(words) atIndex:2];
        } else {
            [enc setBuffer:buf offset:(NSUInteger)a->offset atIndex:0];
            [enc setBytes:&words length:sizeof(words) atIndex:1];
            uint32_t seed = a->seed;
            [enc setBytes:&seed length:sizeof(seed) atIndex:2];
        }
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = now_ns();

        a->wall_ns = t1 - t0;
        double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        a->gpu_ns = (gpu_s > 0.0) ? (uint64_t)(gpu_s * 1e9) : a->wall_ns;
        if (a->mode == 0) {
            uint64_t sum = 0;
            const auto *p = static_cast<const uint64_t *>(partials.contents);
            for (uint32_t i = 0; i < groups; ++i) sum += p[i];
            a->checksum_out = sum;
        } else {
            a->checksum_out = 0;
        }
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_convert_dxil(void *raw) {
    auto *a = static_cast<vm_convert_dxil_args *>(raw);
    if (!a || a->dxil_ptr == 0 || a->dxil_len == 0) return VM_STATUS_INVALID_PARAM;
    auto &api = ir::api();
    if (!api.loaded()) {
        a->status = -1;
        a->out_len = 0;
        return VM_STATUS_SUCCESS;
    }

    const char *entry = a->entry_ptr ? reinterpret_cast<const char *>(a->entry_ptr) : "main";
    ir::Compiler *comp = api.compiler_create();
    if (!comp) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    if (api.compiler_set_target) {
        api.compiler_set_target(comp, 0 /* macOS */, "15.0");
    }
    if (api.compiler_set_entry) {
        api.compiler_set_entry(comp, entry);
    }

    ir::Object *dxil = api.object_from_dxil(
        reinterpret_cast<const uint8_t *>(a->dxil_ptr),
        (size_t)a->dxil_len,
        0 /* IRBytecodeOwnershipNone */
    );
    if (!dxil) {
        api.compiler_destroy(comp);
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }

    ir::Error *err = nullptr;
    ir::Object *out_ir = api.compiler_compile(comp, entry, dxil, &err);
    if (!out_ir) {
        if (err && api.error_destroy) api.error_destroy(err);
        api.object_destroy(dxil);
        api.compiler_destroy(comp);
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }

    uint32_t stage = api.object_get_stage(out_ir);
    a->stage = stage;
    ir::MetalLibBinary *mlib = api.metallib_create();
    bool got = mlib && api.object_get_metallib(out_ir, stage, mlib);
    if (!got) {
        if (mlib) api.metallib_destroy(mlib);
        api.object_destroy(out_ir);
        api.object_destroy(dxil);
        api.compiler_destroy(comp);
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }

    size_t sz = api.metallib_size(mlib);
    a->out_len = (uint64_t)sz;
    if (a->out_ptr == 0 || a->out_cap < sz) {
        a->status = (a->out_ptr == 0) ? 0 : -3;
    } else {
        api.metallib_bytes(mlib, reinterpret_cast<uint8_t *>(a->out_ptr));
        a->status = 0;
        state().dxil_converted.fetch_add(1, std::memory_order_relaxed);
    }

    api.metallib_destroy(mlib);
    api.object_destroy(out_ir);
    api.object_destroy(dxil);
    api.compiler_destroy(comp);
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_stats(void *raw) {
    auto *a = static_cast<vm_stats_args *>(raw);
    if (!a) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    a->active_buffers = s.by_handle.size();
    a->active_bytes = s.active_bytes;
    a->total_registered = s.total_registered;
    a->total_reused = s.total_reused;
    a->residency_commits = s.residency_commits;
    a->dxil_converted = s.dxil_converted.load(std::memory_order_relaxed);
    return VM_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Step 2B-2: Textures, MetalFX Temporal Scaler, MetalFX Frame Interpolator,
// Built-in Metal Optical Flow, and Native CAMetalLayer Presentation
// ---------------------------------------------------------------------------

static id<MTLTexture> make_texture(
    id<MTLDevice> device,
    uint32_t w,
    uint32_t h,
    MTLPixelFormat fmt,
    MTLTextureUsage usage,
    MTLStorageMode storage
) {
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    desc.storageMode = storage;
    desc.usage = usage ? usage
                       : (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                          MTLTextureUsageRenderTarget);
    return [device newTextureWithDescriptor:desc];
}

static NTSTATUS op_create_texture(void *raw) {
    auto *a = static_cast<vm_create_texture_args *>(raw);
    if (!a || a->width == 0 || a->height == 0 || a->pixel_format == 0) {
        return VM_STATUS_INVALID_PARAM;
    }
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    @autoreleasepool {
        MTLStorageMode sm = (a->storage_mode == 1) ? MTLStorageModeShared : MTLStorageModePrivate;
        id<MTLTexture> tex = make_texture(
            s.device,
            a->width,
            a->height,
            (MTLPixelFormat)a->pixel_format,
            (MTLTextureUsage)a->usage,
            sm
        );
        if (!tex) {
            a->handle = 0;
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        uint64_t h = s.next_handle++;
        TextureEntry e;
        e.handle = h;
        e.width = a->width;
        e.height = a->height;
        e.format = (MTLPixelFormat)a->pixel_format;
        e.texture = tex;
        s.textures.emplace(h, e);
        a->handle = h;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_destroy_texture(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    @autoreleasepool {
        s.textures.erase(a->handle);
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_upload_texture(void *raw) {
    auto *a = static_cast<vm_texture_io_args *>(raw);
    if (!a || a->texture_handle == 0 || a->bytes_per_row == 0) return VM_STATUS_INVALID_PARAM;
    if (a->buffer_handle == 0 && a->cpu_ptr == 0) return VM_STATUS_INVALID_PARAM;

    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    auto tit = s.textures.find(a->texture_handle);
    if (tit == s.textures.end() || !tit->second.texture) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    id<MTLTexture> tex = tit->second.texture;
    uint32_t w = a->width ? a->width : tit->second.width;
    uint32_t h = a->height ? a->height : tit->second.height;
    size_t total_bytes = (size_t)a->bytes_per_row * (size_t)h;

    @autoreleasepool {
        id<MTLBuffer> src_buf = nil;
        NSUInteger src_off = 0;
        if (a->buffer_handle != 0) {
            auto bit = s.by_handle.find(a->buffer_handle);
            if (bit == s.by_handle.end() || a->offset + total_bytes > bit->second.length) {
                a->status = -3;
                return VM_STATUS_SUCCESS;
            }
            src_buf = bit->second.buffer;
            src_off = (NSUInteger)a->offset;
        } else {
            src_buf = s.ensure_staging_locked(total_bytes);
            if (!src_buf) {
                a->status = -4;
                return VM_STATUS_SUCCESS;
            }
            std::memcpy(src_buf.contents, reinterpret_cast<const void *>(a->cpu_ptr), total_bytes);
            src_off = 0;
        }

        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src_buf
                sourceOffset:src_off
           sourceBytesPerRow:a->bytes_per_row
         sourceBytesPerImage:total_bytes
                  sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:tex
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        a->status = (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -5;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_readback_texture(void *raw) {
    auto *a = static_cast<vm_texture_io_args *>(raw);
    if (!a || a->texture_handle == 0 || a->bytes_per_row == 0) return VM_STATUS_INVALID_PARAM;
    if (a->buffer_handle == 0 && a->cpu_ptr == 0) return VM_STATUS_INVALID_PARAM;

    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    auto tit = s.textures.find(a->texture_handle);
    if (tit == s.textures.end() || !tit->second.texture) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    id<MTLTexture> tex = tit->second.texture;
    uint32_t w = a->width ? a->width : tit->second.width;
    uint32_t h = a->height ? a->height : tit->second.height;
    size_t total_bytes = (size_t)a->bytes_per_row * (size_t)h;

    @autoreleasepool {
        id<MTLBuffer> dst_buf = nil;
        NSUInteger dst_off = 0;
        bool copy_to_cpu = false;
        if (a->buffer_handle != 0) {
            auto bit = s.by_handle.find(a->buffer_handle);
            if (bit == s.by_handle.end() || a->offset + total_bytes > bit->second.length) {
                a->status = -3;
                return VM_STATUS_SUCCESS;
            }
            dst_buf = bit->second.buffer;
            dst_off = (NSUInteger)a->offset;
        } else {
            dst_buf = s.ensure_staging_locked(total_bytes);
            if (!dst_buf) {
                a->status = -4;
                return VM_STATUS_SUCCESS;
            }
            dst_off = 0;
            copy_to_cpu = true;
        }

        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:tex
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                     toBuffer:dst_buf
            destinationOffset:dst_off
       destinationBytesPerRow:a->bytes_per_row
     destinationBytesPerImage:total_bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            a->status = -5;
            return VM_STATUS_SUCCESS;
        }
        if (copy_to_cpu) {
            std::memcpy(reinterpret_cast<void *>(a->cpu_ptr), dst_buf.contents, total_bytes);
        }
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_scaler_create(void *raw) {
    auto *a = static_cast<vm_scaler_create_args *>(raw);
    if (!a || a->input_width == 0 || a->input_height == 0 ||
        a->output_width == 0 || a->output_height == 0) {
        return VM_STATUS_INVALID_PARAM;
    }
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked() || !s.has_metalfx) {
        a->status = -1;
        return VM_STATUS_SUCCESS;
    }
    @autoreleasepool {
        if (@available(macOS 13.0, *)) {
            MTLFXTemporalScalerDescriptor *desc = [[MTLFXTemporalScalerDescriptor alloc] init];
            desc.colorTextureFormat = (MTLPixelFormat)a->color_format;
            desc.depthTextureFormat = (MTLPixelFormat)a->depth_format;
            desc.motionTextureFormat = (MTLPixelFormat)a->motion_format;
            desc.outputTextureFormat = (MTLPixelFormat)a->output_format;
            desc.inputWidth = a->input_width;
            desc.inputHeight = a->input_height;
            desc.outputWidth = a->output_width;
            desc.outputHeight = a->output_height;
            desc.autoExposureEnabled = (a->flags & 1u) != 0;
            desc.requiresSynchronousInitialization = (a->flags & 2u) != 0;
            desc.inputContentPropertiesEnabled = YES;
            float min_s = 1.0f;
            float max_s = 3.0f;
            if (@available(macOS 14.0, *)) {
                min_s = [MTLFXTemporalScalerDescriptor supportedInputContentMinScaleForDevice:s.device];
                max_s = [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:s.device];
            }
            desc.inputContentMinScale = min_s;
            desc.inputContentMaxScale = max_s;
            if (a->reactive_format != 0) {
                if (@available(macOS 14.4, *)) {
                    desc.reactiveMaskTextureEnabled = YES;
                    desc.reactiveMaskTextureFormat = (MTLPixelFormat)a->reactive_format;
                }
            }
            id<MTLFXTemporalScaler> scaler = [desc newTemporalScalerWithDevice:s.device];
            if (!scaler) {
                a->handle = 0;
                a->status = -2;
                return VM_STATUS_SUCCESS;
            }
            uint64_t h = s.next_handle++;
            ScalerEntry entry;
            entry.handle = h;
            entry.input_width = a->input_width;
            entry.input_height = a->input_height;
            entry.output_width = a->output_width;
            entry.output_height = a->output_height;
            entry.scaler = scaler;
            s.scalers.emplace(h, entry);
            a->handle = h;
            a->status = 0;
            return VM_STATUS_SUCCESS;
        }
        a->status = -1;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_scaler_encode(void *raw) {
    auto *a = static_cast<vm_scaler_encode_args *>(raw);
    if (!a || a->scaler_handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto sit = s.scalers.find(a->scaler_handle);
    if (sit == s.scalers.end() || !sit->second.scaler) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    auto find_tex = [&](uint64_t h) -> id<MTLTexture> {
        if (h == 0) return nil;
        auto it = s.textures.find(h);
        return (it != s.textures.end()) ? it->second.texture : nil;
    };
    id<MTLTexture> color = find_tex(a->color_tex);
    id<MTLTexture> depth = find_tex(a->depth_tex);
    id<MTLTexture> motion = find_tex(a->motion_tex);
    id<MTLTexture> reactive = find_tex(a->reactive_tex);
    id<MTLTexture> output = find_tex(a->output_tex);
    if (!color || !depth || !motion || !output) {
        a->status = -3;
        return VM_STATUS_SUCCESS;
    }

    @autoreleasepool {
        id<MTLFXTemporalScaler> sc = sit->second.scaler;
        sc.colorTexture = color;
        sc.depthTexture = depth;
        sc.motionTexture = motion;
        sc.outputTexture = output;
        if (reactive) {
            sc.reactiveMaskTexture = reactive;
        }
        sc.inputContentWidth = a->input_content_width ? a->input_content_width : sit->second.input_width;
        sc.inputContentHeight = a->input_content_height ? a->input_content_height : sit->second.input_height;
        sc.jitterOffsetX = a->jitter_x;
        sc.jitterOffsetY = a->jitter_y;
        sc.motionVectorScaleX = a->mv_scale_x;
        sc.motionVectorScaleY = a->mv_scale_y;
        sc.preExposure = (a->pre_exposure > 0.0f) ? a->pre_exposure : 1.0f;
        sc.reset = (a->flags & 1u) != 0;
        sc.depthReversed = (a->flags & 2u) != 0;

        uint64_t t0 = now_ns();
        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        [sc encodeToCommandBuffer:cb];
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = now_ns();

        a->wall_ns = t1 - t0;
        double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        a->gpu_ns = (gpu_s > 0.0) ? (uint64_t)(gpu_s * 1e9) : a->wall_ns;
        a->status = (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -4;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_scaler_destroy(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    @autoreleasepool {
        s.scalers.erase(a->handle);
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_interp_create(void *raw) {
    auto *a = static_cast<vm_interp_create_args *>(raw);
    if (!a || a->output_width == 0 || a->output_height == 0) {
        return VM_STATUS_INVALID_PARAM;
    }
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked() || !s.init_kernels_locked() || !s.has_frame_interp) {
        a->status = -1;
        return VM_STATUS_SUCCESS;
    }
    @autoreleasepool {
        if (@available(macOS 26.0, *)) {
            uint32_t in_w = a->input_width ? a->input_width : std::max(1u, a->output_width / 2u);
            uint32_t in_h = a->input_height ? a->input_height : std::max(1u, a->output_height / 2u);
            MTLPixelFormat color_fmt = (MTLPixelFormat)(a->color_format ? a->color_format : MTLPixelFormatRGBA8Unorm);
            MTLPixelFormat out_fmt = (MTLPixelFormat)(a->output_format ? a->output_format : color_fmt);
            MTLPixelFormat depth_fmt = (MTLPixelFormat)(a->depth_format ? a->depth_format : MTLPixelFormatR32Float);
            MTLPixelFormat motion_fmt = (MTLPixelFormat)(a->motion_format ? a->motion_format : MTLPixelFormatRG16Float);
            MTLPixelFormat ui_fmt = (MTLPixelFormat)a->ui_format;

            MTLFXFrameInterpolatorDescriptor *desc = [[MTLFXFrameInterpolatorDescriptor alloc] init];
            desc.colorTextureFormat = color_fmt;
            desc.outputTextureFormat = out_fmt;
            desc.depthTextureFormat = depth_fmt;
            desc.motionTextureFormat = motion_fmt;
            if (ui_fmt != MTLPixelFormatInvalid) {
                desc.uiTextureFormat = ui_fmt;
            }
            desc.inputWidth = in_w;
            desc.inputHeight = in_h;
            desc.outputWidth = a->output_width;
            desc.outputHeight = a->output_height;
            if (@available(macOS 27.0, *)) {
                desc.requiresPrevColorTexture = YES;
            }

            id<MTLFXFrameInterpolator> interp = [desc newFrameInterpolatorWithDevice:s.device];
            if (!interp) {
                a->handle = 0;
                a->status = -2;
                return VM_STATUS_SUCCESS;
            }

            MTLTextureUsage color_u = interp.colorTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            MTLTextureUsage motion_u = interp.motionTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            MTLTextureUsage depth_u = interp.depthTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> prev_tex = make_texture(s.device, a->output_width, a->output_height, color_fmt, color_u, MTLStorageModePrivate);
            id<MTLTexture> flow_tex = make_texture(s.device, in_w, in_h, motion_fmt, motion_u, MTLStorageModePrivate);
            id<MTLTexture> depth_tex = make_texture(s.device, in_w, in_h, depth_fmt, depth_u, MTLStorageModePrivate);
            id<MTLTexture> resampled_ui = nil;
            if (ui_fmt != MTLPixelFormatInvalid) {
                MTLTextureUsage ui_u = interp.uiTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                resampled_ui = make_texture(s.device, a->output_width, a->output_height, ui_fmt, ui_u, MTLStorageModePrivate);
            }
            if (!prev_tex || !flow_tex || !depth_tex) {
                a->handle = 0;
                a->status = -3;
                return VM_STATUS_SUCCESS;
            }

            uint64_t h = s.next_handle++;
            InterpEntry entry;
            entry.handle = h;
            entry.input_width = in_w;
            entry.input_height = in_h;
            entry.output_width = a->output_width;
            entry.output_height = a->output_height;
            entry.color_format = color_fmt;
            entry.ui_format = ui_fmt;
            entry.interpolator = interp;
            entry.prev_color = prev_tex;
            entry.flow_motion = flow_tex;
            entry.const_depth = depth_tex;
            entry.resampled_ui = resampled_ui;
            entry.has_prev = false;
            s.interps.emplace(h, entry);

            a->handle = h;
            a->status = 0;
            return VM_STATUS_SUCCESS;
        }
        a->status = -1;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_interp_encode(void *raw) {
    auto *a = static_cast<vm_interp_encode_args *>(raw);
    if (!a || a->interp_handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto iit = s.interps.find(a->interp_handle);
    if (iit == s.interps.end() || !iit->second.interpolator) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    auto find_tex = [&](uint64_t h) -> id<MTLTexture> {
        if (h == 0) return nil;
        auto it = s.textures.find(h);
        return (it != s.textures.end()) ? it->second.texture : nil;
    };
    InterpEntry &entry = iit->second;
    id<MTLTexture> color = find_tex(a->color_tex);
    id<MTLTexture> explicit_prev = find_tex(a->prev_color_tex);
    id<MTLTexture> depth = find_tex(a->depth_tex);
    id<MTLTexture> motion = find_tex(a->motion_tex);
    id<MTLTexture> ui = find_tex(a->ui_tex);
    id<MTLTexture> output = find_tex(a->output_tex);
    if (!color || !output) {
        a->status = -3;
        return VM_STATUS_SUCCESS;
    }

    @autoreleasepool {
        if (@available(macOS 26.0, *)) {
            id<MTLFXFrameInterpolator> interp = entry.interpolator;
            bool force_reset = (a->flags & 1u) != 0;
            bool skip_prev_update = (a->flags & 8u) != 0;
            id<MTLTexture> prev = explicit_prev ? explicit_prev : entry.prev_color;
            bool have_prev = explicit_prev != nil || (entry.has_prev && !force_reset);

            uint64_t t0 = now_ns();
            id<MTLCommandBuffer> cb = [s.queue commandBuffer];

            if (!have_prev) {
                // First frame / scene reset: seed internal prevColor so next frame interpolates.
                if (!skip_prev_update) {
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromTexture:color toTexture:entry.prev_color];
                    [blit endEncoding];
                    entry.has_prev = true;
                }
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t t1 = now_ns();
                a->interpolated = 0;
                a->wall_ns = t1 - t0;
                a->gpu_ns = 0;
                a->status = (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -4;
                return VM_STATUS_SUCCESS;
            }

            // If no engine motion/depth supplied, compute dense optical flow in-place on GPU.
            bool use_flow = (!motion || !depth);
            if (use_flow) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:s.optical_flow_pso];
                [enc setTexture:color atIndex:0];
                [enc setTexture:prev atIndex:1];
                [enc setTexture:entry.flow_motion atIndex:2];
                [enc setTexture:entry.const_depth atIndex:3];
                MTLSize tg = MTLSizeMake(16, 16, 1);
                MTLSize grid = MTLSizeMake(
                    (entry.input_width + 15) / 16,
                    (entry.input_height + 15) / 16,
                    1
                );
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                motion = entry.flow_motion;
                depth = entry.const_depth;
            }

            // Resample arbitrary-size Plane 1 UI (e.g. Series S 3840x2160 -> 1920x1080).
            id<MTLTexture> ui_for_interp = nil;
            if (ui && entry.ui_format != MTLPixelFormatInvalid) {
                if (ui.width == entry.output_width && ui.height == entry.output_height &&
                    ui.pixelFormat == entry.ui_format) {
                    ui_for_interp = ui;
                } else if (entry.resampled_ui) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:s.resample_ui_pso];
                    [enc setTexture:ui atIndex:0];
                    [enc setTexture:entry.resampled_ui atIndex:1];
                    MTLSize tg = MTLSizeMake(16, 16, 1);
                    MTLSize grid = MTLSizeMake(
                        (entry.output_width + 15) / 16,
                        (entry.output_height + 15) / 16,
                        1
                    );
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                    ui_for_interp = entry.resampled_ui;
                }
            }

            interp.colorTexture = color;
            interp.prevColorTexture = prev;
            interp.depthTexture = depth;
            interp.motionTexture = motion;
            interp.outputTexture = output;
            interp.uiTexture = ui_for_interp;
            interp.uiTextureComposited = (a->flags & 4u) != 0;
            interp.motionVectorScaleX = use_flow ? 1.0f : (a->mv_scale_x != 0.0f ? a->mv_scale_x : 1.0f);
            interp.motionVectorScaleY = use_flow ? 1.0f : (a->mv_scale_y != 0.0f ? a->mv_scale_y : 1.0f);
            interp.jitterOffsetX = a->jitter_x;
            interp.jitterOffsetY = a->jitter_y;
            interp.deltaTime = (a->delta_time > 0.0f) ? a->delta_time : (1.0f / 30.0f);
            interp.nearPlane = (a->near_plane > 0.0f) ? a->near_plane : 0.1f;
            interp.farPlane = (a->far_plane > 0.0f) ? a->far_plane : 1000.0f;
            interp.fieldOfView = (a->fov_degrees > 1.0f) ? a->fov_degrees : 60.0f;
            interp.aspectRatio = (float)entry.output_width / (float)entry.output_height;
            interp.depthReversed = use_flow ? NO : ((a->flags & 2u) != 0);
            interp.shouldResetHistory = force_reset;

            [interp encodeToCommandBuffer:cb];

            if (!skip_prev_update && !explicit_prev) {
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromTexture:color toTexture:entry.prev_color];
                [blit endEncoding];
                entry.has_prev = true;
            }

            [cb commit];
            [cb waitUntilCompleted];
            uint64_t t1 = now_ns();

            a->wall_ns = t1 - t0;
            double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
            a->gpu_ns = (gpu_s > 0.0) ? (uint64_t)(gpu_s * 1e9) : a->wall_ns;
            a->interpolated = (cb.status == MTLCommandBufferStatusCompleted) ? 1 : 0;
            a->status = (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -4;
            return VM_STATUS_SUCCESS;
        }
        a->status = -1;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_interp_destroy(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    @autoreleasepool {
        s.interps.erase(a->handle);
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_surface_attach(void *raw) {
    auto *a = static_cast<vm_surface_attach_args *>(raw);
    if (!a || a->width == 0 || a->height == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    id<MTLDevice> dev = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.init_locked() || !s.init_kernels_locked()) {
            a->status = -1;
            return VM_STATUS_SUCCESS;
        }
        dev = s.device;
    }

    __block VictoryMetalOverlayView *overlay = nil;
    __block CAMetalLayer *metal_layer = nil;
    __block int attach_err = -2;
    uint32_t req_w = a->width;
    uint32_t req_h = a->height;
    MTLPixelFormat pf = (MTLPixelFormat)(a->pixel_format ? a->pixel_format : MTLPixelFormatBGRA8Unorm);
    bool vsync = (a->flags & 1u) != 0;

    bool ran = run_on_main_bounded(^{
        if (!NSApp) {
            attach_err = -3;
            return;
        }
        NSWindow *target = nil;
        for (NSWindow *win in [NSApp windows]) {
            if (!win.visible || !win.contentView) continue;
            NSRect f = win.contentView.bounds;
            if (f.size.width >= 64 && f.size.height >= 64) {
                target = win;
                break;
            }
        }
        if (!target) {
            attach_err = -4;
            return;
        }
        NSView *content = target.contentView;
        VictoryMetalOverlayView *v = [[VictoryMetalOverlayView alloc] initWithFrame:content.bounds];
        v.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        v.wantsLayer = YES;
        CAMetalLayer *ml = v.metalLayer;
        if (!ml && [v.layer isKindOfClass:[CAMetalLayer class]]) {
            ml = (CAMetalLayer *)v.layer;
        }
        if (!ml) {
            attach_err = -5;
            return;
        }
        ml.device = dev;
        ml.pixelFormat = pf;
        ml.framebufferOnly = NO;
        ml.maximumDrawableCount = 3;
        ml.displaySyncEnabled = vsync;
        ml.drawableSize = CGSizeMake(req_w, req_h);
        [content addSubview:v positioned:NSWindowAbove relativeTo:nil];
        overlay = v;
        metal_layer = ml;
        attach_err = 0;
    }, 250);

    if (!ran || attach_err != 0 || !metal_layer) {
        a->handle = 0;
        a->status = ran ? attach_err : -6;
        return VM_STATUS_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(s.mu);
    uint64_t h = s.next_handle++;
    SurfaceEntry entry;
    entry.handle = h;
    entry.width = req_w;
    entry.height = req_h;
    entry.view = overlay;
    entry.layer = metal_layer;
    s.surfaces.emplace(h, entry);
    a->handle = h;
    a->status = 0;
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_surface_present(void *raw) {
    auto *a = static_cast<vm_surface_present_args *>(raw);
    if (!a || a->surface_handle == 0 || a->color_tex == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    CAMetalLayer *layer = nil;
    id<MTLTexture> color = nil;
    id<MTLTexture> ui = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> comp_pso = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        auto sit = s.surfaces.find(a->surface_handle);
        if (sit == s.surfaces.end() || !sit->second.layer) {
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        auto cit = s.textures.find(a->color_tex);
        if (cit == s.textures.end() || !cit->second.texture) {
            a->status = -3;
            return VM_STATUS_SUCCESS;
        }
        layer = sit->second.layer;
        color = cit->second.texture;
        if (a->ui_tex != 0) {
            auto uit = s.textures.find(a->ui_tex);
            if (uit != s.textures.end()) ui = uit->second.texture;
        }
        queue = s.queue;
        comp_pso = s.composite_present_pso;
    }

    @autoreleasepool {
        uint64_t t0 = now_ns();
        // Update drawableSize cleanly in-place if the texture resolution changed
        // (avoids destroying/recreating the NSView while drawables are in flight).
        CGSize want = CGSizeMake(color.width, color.height);
        if (layer.drawableSize.width != want.width || layer.drawableSize.height != want.height) {
            layer.drawableSize = want;
        }
        id<CAMetalDrawable> drawable = [layer nextDrawable];
        if (!drawable || !drawable.texture) {
            a->status = -4;
            return VM_STATUS_SUCCESS;
        }
        id<MTLTexture> dst = drawable.texture;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pso];
        [enc setTexture:color atIndex:0];
        [enc setTexture:(ui ? ui : color) atIndex:1];
        [enc setTexture:dst atIndex:2];
        uint32_t has_ui = ui ? 1u : 0u;
        [enc setBytes:&has_ui length:sizeof(has_ui) atIndex:0];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((dst.width + 15) / 16, (dst.height + 15) / 16, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        // Present directly without afterMinimumDuration so the 3-drawable pool
        // never stalls during fast loading screens or mode transitions.
        (void)a->after_min_duration_ns;
        [cb presentDrawable:drawable];
        [cb commit];
        if ((a->flags & 1u) != 0) {
            [cb waitUntilCompleted];
        }
        uint64_t t1 = now_ns();
        a->wall_ns = t1 - t0;
        double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        a->gpu_ns = (gpu_s > 0.0) ? (uint64_t)(gpu_s * 1e9) : 0;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_surface_destroy(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    VictoryMetalOverlayView *view = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        auto it = s.surfaces.find(a->handle);
        if (it == s.surfaces.end()) return VM_STATUS_SUCCESS;
        view = it->second.view;
        s.surfaces.erase(it);
    }
    if (view) {
        run_on_main_bounded(^{
            [view removeFromSuperview];
        }, 250);
    }
    return VM_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Step 2B-3: MTLBinaryArchive, Direct Compute/Render Pipelines, and
// Batched MTLCommandBuffer Submission
// ---------------------------------------------------------------------------

static bool dxil_to_metallib_vec(
    const uint8_t *dxil_bytes,
    size_t dxil_len,
    const char *entry,
    std::vector<uint8_t> &out
) {
    auto &api = ir::api();
    if (!api.loaded() || !dxil_bytes || dxil_len == 0) return false;
    ir::Compiler *comp = api.compiler_create();
    if (!comp) return false;
    if (api.compiler_set_target) api.compiler_set_target(comp, 0, "15.0");
    if (api.compiler_set_entry) api.compiler_set_entry(comp, entry ? entry : "main");
    ir::Object *dxil = api.object_from_dxil(dxil_bytes, dxil_len, 0);
    if (!dxil) {
        api.compiler_destroy(comp);
        return false;
    }
    ir::Error *err = nullptr;
    ir::Object *out_ir = api.compiler_compile(comp, entry ? entry : "main", dxil, &err);
    if (!out_ir) {
        if (err && api.error_destroy) api.error_destroy(err);
        api.object_destroy(dxil);
        api.compiler_destroy(comp);
        return false;
    }
    uint32_t stage = api.object_get_stage(out_ir);
    ir::MetalLibBinary *mlib = api.metallib_create();
    bool ok = mlib && api.object_get_metallib(out_ir, stage, mlib);
    if (ok) {
        size_t sz = api.metallib_size(mlib);
        out.resize(sz);
        api.metallib_bytes(mlib, out.data());
        state().dxil_converted.fetch_add(1, std::memory_order_relaxed);
    }
    if (mlib) api.metallib_destroy(mlib);
    api.object_destroy(out_ir);
    api.object_destroy(dxil);
    api.compiler_destroy(comp);
    return ok;
}

static id<MTLLibrary> compile_library(
    id<MTLDevice> device,
    const uint8_t *bytes,
    size_t len,
    const char *entry,
    uint32_t source_kind
) {
    if (!bytes || len == 0) return nil;
    NSError *err = nil;
    if (source_kind == 2) {
        NSString *src = [[NSString alloc] initWithBytes:bytes
                                                 length:len
                                               encoding:NSUTF8StringEncoding];
        if (!src) return nil;
        return [device newLibraryWithSource:src options:nil error:&err];
    }
    std::vector<uint8_t> mlib_storage;
    const uint8_t *mlib_ptr = bytes;
    size_t mlib_len = len;
    if (source_kind == 0) {
        if (!dxil_to_metallib_vec(bytes, len, entry, mlib_storage)) return nil;
        mlib_ptr = mlib_storage.data();
        mlib_len = mlib_storage.size();
    }
    dispatch_data_t data = dispatch_data_create(
        mlib_ptr,
        mlib_len,
        dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT
    );
    return [device newLibraryWithData:data error:&err];
}

static NTSTATUS op_archive_open(void *raw) {
    auto *a = static_cast<vm_archive_open_args *>(raw);
    if (!a) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    @autoreleasepool {
        MTLBinaryArchiveDescriptor *desc = [[MTLBinaryArchiveDescriptor alloc] init];
        bool loaded = false;
        std::string path_str;
        if (a->path_ptr != 0) {
            const char *cpath = reinterpret_cast<const char *>(a->path_ptr);
            if (cpath && *cpath) {
                path_str = cpath;
                NSString *ns = [NSString stringWithUTF8String:cpath];
                if ([[NSFileManager defaultManager] fileExistsAtPath:ns]) {
                    desc.url = [NSURL fileURLWithPath:ns];
                    loaded = true;
                }
            }
        }
        NSError *err = nil;
        id<MTLBinaryArchive> arch = [s.device newBinaryArchiveWithDescriptor:desc error:&err];
        if (!arch && loaded) {
            // Corrupt or incompatible archive on disk: start fresh cleanly.
            desc.url = nil;
            loaded = false;
            err = nil;
            arch = [s.device newBinaryArchiveWithDescriptor:desc error:&err];
        }
        if (!arch) {
            a->handle = 0;
            a->loaded_existing = 0;
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        uint64_t h = s.next_handle++;
        ArchiveEntry e;
        e.handle = h;
        e.archive = arch;
        e.path = path_str;
        s.archives.emplace(h, e);
        a->handle = h;
        a->loaded_existing = loaded ? 1u : 0u;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_archive_save(void *raw) {
    auto *a = static_cast<vm_archive_save_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.archives.find(a->handle);
    if (it == s.archives.end() || !it->second.archive) {
        a->status = -2;
        return VM_STATUS_SUCCESS;
    }
    @autoreleasepool {
        std::string target = it->second.path;
        if (a->path_ptr != 0) {
            const char *cp = reinterpret_cast<const char *>(a->path_ptr);
            if (cp && *cp) target = cp;
        }
        if (target.empty()) {
            a->status = -3;
            return VM_STATUS_SUCCESS;
        }
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:target.c_str()]];
        NSError *err = nil;
        BOOL ok = [it->second.archive serializeToURL:url error:&err];
        a->status = ok ? 0 : -4;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_archive_close(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    @autoreleasepool {
        s.archives.erase(a->handle);
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_pipeline_create_compute(void *raw) {
    auto *a = static_cast<vm_pipeline_compute_args *>(raw);
    if (!a || a->bytes_ptr == 0 || a->bytes_len == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    id<MTLDevice> dev = nil;
    id<MTLBinaryArchive> arch = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.init_locked()) {
            a->status = -1;
            return VM_STATUS_UNSUCCESSFUL;
        }
        dev = s.device;
        if (a->archive_handle != 0) {
            auto ait = s.archives.find(a->archive_handle);
            if (ait != s.archives.end()) arch = ait->second.archive;
        }
    }
    // Compile DXIL -> MetalLib and PSO OUTSIDE s.mu so presentation and
    // frame interpolation never stall behind background shader compilation.
    id<MTLComputePipelineState> pso = nil;
    uint32_t max_tg = 0;
    uint32_t simd_w = 0;
    @autoreleasepool {
        const char *entry = a->entry_ptr ? reinterpret_cast<const char *>(a->entry_ptr) : "main";
        id<MTLLibrary> lib = compile_library(
            dev,
            reinterpret_cast<const uint8_t *>(a->bytes_ptr),
            (size_t)a->bytes_len,
            entry,
            a->source_kind
        );
        if (!lib) {
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        NSString *fn_name = [NSString stringWithUTF8String:entry];
        id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
        if (!fn && lib.functionNames.count > 0) {
            fn = [lib newFunctionWithName:lib.functionNames[0]];
        }
        if (!fn) {
            a->status = -3;
            return VM_STATUS_SUCCESS;
        }
        MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = fn;
        if (arch) {
            desc.binaryArchives = @[arch];
        }
        NSError *err = nil;
        pso = [dev newComputePipelineStateWithDescriptor:desc
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:&err];
        if (!pso) {
            a->status = -4;
            return VM_STATUS_SUCCESS;
        }
        if (arch) {
            std::lock_guard<std::mutex> lock(s.mu);
            NSError *aerr = nil;
            [arch addComputePipelineFunctionsWithDescriptor:desc error:&aerr];
        }
        max_tg = (uint32_t)pso.maxTotalThreadsPerThreadgroup;
        simd_w = (uint32_t)pso.threadExecutionWidth;
    }
    {
        std::lock_guard<std::mutex> lock(s.mu);
        uint64_t h = s.next_handle++;
        ComputePsoEntry e;
        e.handle = h;
        e.pso = pso;
        s.compute_psos.emplace(h, e);
        a->handle = h;
        a->max_threads_per_tg = max_tg;
        a->simd_width = simd_w;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_pipeline_create_render(void *raw) {
    auto *a = static_cast<vm_pipeline_render_args *>(raw);
    if (!a || a->vs_ptr == 0 || a->vs_len == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    @autoreleasepool {
        const char *vs_entry = a->vs_entry_ptr ? reinterpret_cast<const char *>(a->vs_entry_ptr) : "vs_main";
        const char *fs_entry = a->fs_entry_ptr ? reinterpret_cast<const char *>(a->fs_entry_ptr) : "fs_main";
        id<MTLLibrary> vs_lib = compile_library(
            s.device,
            reinterpret_cast<const uint8_t *>(a->vs_ptr),
            (size_t)a->vs_len,
            vs_entry,
            a->source_kind
        );
        if (!vs_lib) {
            a->status = -2;
            return VM_STATUS_SUCCESS;
        }
        id<MTLFunction> vs_fn = [vs_lib newFunctionWithName:[NSString stringWithUTF8String:vs_entry]];
        if (!vs_fn) {
            a->status = -3;
            return VM_STATUS_SUCCESS;
        }
        id<MTLFunction> fs_fn = nil;
        if (a->fs_ptr != 0 && a->fs_len != 0) {
            id<MTLLibrary> fs_lib = (a->fs_ptr == a->vs_ptr && a->fs_len == a->vs_len)
                ? vs_lib
                : compile_library(
                    s.device,
                    reinterpret_cast<const uint8_t *>(a->fs_ptr),
                    (size_t)a->fs_len,
                    fs_entry,
                    a->source_kind
                  );
            if (!fs_lib) {
                a->status = -4;
                return VM_STATUS_SUCCESS;
            }
            fs_fn = [fs_lib newFunctionWithName:[NSString stringWithUTF8String:fs_entry]];
            if (!fs_fn) {
                a->status = -5;
                return VM_STATUS_SUCCESS;
            }
        }
        MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vs_fn;
        desc.fragmentFunction = fs_fn;
        MTLPixelFormat cf = (MTLPixelFormat)(a->color_format ? a->color_format : MTLPixelFormatRGBA8Unorm);
        desc.colorAttachments[0].pixelFormat = cf;
        if ((a->flags & 1u) != 0) {
            desc.colorAttachments[0].blendingEnabled = YES;
            desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
        if (a->depth_format != 0) {
            desc.depthAttachmentPixelFormat = (MTLPixelFormat)a->depth_format;
        }
        id<MTLBinaryArchive> arch = nil;
        if (a->archive_handle != 0) {
            auto ait = s.archives.find(a->archive_handle);
            if (ait != s.archives.end()) {
                arch = ait->second.archive;
                desc.binaryArchives = @[arch];
            }
        }
        NSError *err = nil;
        id<MTLRenderPipelineState> pso = [s.device newRenderPipelineStateWithDescriptor:desc error:&err];
        if (!pso) {
            a->status = -6;
            return VM_STATUS_SUCCESS;
        }
        if (arch) {
            NSError *aerr = nil;
            [arch addRenderPipelineFunctionsWithDescriptor:desc error:&aerr];
        }
        uint64_t h = s.next_handle++;
        RenderPsoEntry e;
        e.handle = h;
        e.pso = pso;
        s.render_psos.emplace(h, e);
        a->handle = h;
        a->status = 0;
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_pipeline_destroy(void *raw) {
    auto *a = static_cast<vm_destroy_handle_args *>(raw);
    if (!a || a->handle == 0) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    @autoreleasepool {
        s.compute_psos.erase(a->handle);
        s.render_psos.erase(a->handle);
    }
    return VM_STATUS_SUCCESS;
}

static inline uint64_t read_u64_words(const uint32_t *w, size_t &i) {
    uint64_t lo = w[i++];
    uint64_t hi = w[i++];
    return lo | (hi << 32);
}

static inline float read_f32_word(const uint32_t *w, size_t &i) {
    uint32_t bits = w[i++];
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static NTSTATUS op_submit_commands(void *raw) {
    auto *a = static_cast<vm_submit_commands_args *>(raw);
    if (!a || (a->word_count > 0 && a->stream_ptr == 0)) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.init_locked()) {
        a->status = -1;
        return VM_STATUS_UNSUCCESSFUL;
    }
    s.commit_residency_locked();

    const uint32_t *words = reinterpret_cast<const uint32_t *>(a->stream_ptr);
    const size_t n = a->word_count;

    auto get_buf = [&](uint64_t h) -> id<MTLBuffer> {
        auto it = s.by_handle.find(h);
        return (it != s.by_handle.end()) ? it->second.buffer : nil;
    };
    auto get_tex = [&](uint64_t h) -> id<MTLTexture> {
        auto it = s.textures.find(h);
        return (it != s.textures.end()) ? it->second.texture : nil;
    };

    uint32_t draws = 0;
    uint32_t dispatches = 0;
    int32_t err_code = 0;

    @autoreleasepool {
        uint64_t t_wall0 = now_ns();
        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> cenc = nil;
        id<MTLRenderCommandEncoder> renc = nil;

        auto end_active = [&]() {
            if (cenc) {
                [cenc endEncoding];
                cenc = nil;
            }
            if (renc) {
                [renc endEncoding];
                renc = nil;
            }
        };

        size_t i = 0;
        while (i < n && err_code == 0) {
            uint32_t op = words[i++];
            switch (op) {
                case CMD_COMPUTE_BEGIN: {
                    end_active();
                    cenc = [cb computeCommandEncoder];
                    break;
                }
                case CMD_COMPUTE_END: {
                    if (cenc) {
                        [cenc endEncoding];
                        cenc = nil;
                    }
                    break;
                }
                case CMD_SET_COMPUTE_PSO: {
                    if (i + 2 > n) { err_code = -10; break; }
                    uint64_t h = read_u64_words(words, i);
                    auto it = s.compute_psos.find(h);
                    if (it == s.compute_psos.end() || !cenc) { err_code = -11; break; }
                    [cenc setComputePipelineState:it->second.pso];
                    break;
                }
                case CMD_SET_COMPUTE_BUFFER: {
                    if (i + 5 > n) { err_code = -10; break; }
                    uint32_t slot = words[i++];
                    uint64_t bh = read_u64_words(words, i);
                    uint64_t off = read_u64_words(words, i);
                    id<MTLBuffer> buf = get_buf(bh);
                    if (!buf || !cenc) { err_code = -12; break; }
                    [cenc setBuffer:buf offset:(NSUInteger)off atIndex:slot];
                    break;
                }
                case CMD_SET_COMPUTE_BYTES: {
                    if (i + 2 > n) { err_code = -10; break; }
                    uint32_t slot = words[i++];
                    uint32_t byte_len = words[i++];
                    size_t wlen = (byte_len + 3) / 4;
                    if (i + wlen > n || !cenc) { err_code = -13; break; }
                    [cenc setBytes:&words[i] length:byte_len atIndex:slot];
                    i += wlen;
                    break;
                }
                case CMD_SET_COMPUTE_TEXTURE: {
                    if (i + 3 > n) { err_code = -10; break; }
                    uint32_t slot = words[i++];
                    uint64_t th = read_u64_words(words, i);
                    id<MTLTexture> tex = get_tex(th);
                    if (!tex || !cenc) { err_code = -14; break; }
                    [cenc setTexture:tex atIndex:slot];
                    break;
                }
                case CMD_DISPATCH_TG: {
                    if (i + 6 > n || !cenc) { err_code = -15; break; }
                    uint32_t gx = words[i++], gy = words[i++], gz = words[i++];
                    uint32_t tx = words[i++], ty = words[i++], tz = words[i++];
                    [cenc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
                         threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
                    dispatches++;
                    break;
                }
                case CMD_DISPATCH_INDIRECT: {
                    if (i + 7 > n || !cenc) { err_code = -16; break; }
                    uint64_t bh = read_u64_words(words, i);
                    uint64_t off = read_u64_words(words, i);
                    uint32_t tx = words[i++], ty = words[i++], tz = words[i++];
                    id<MTLBuffer> buf = get_buf(bh);
                    if (!buf) { err_code = -16; break; }
                    [cenc dispatchThreadgroupsWithIndirectBuffer:buf
                                            indirectBufferOffset:(NSUInteger)off
                                           threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
                    dispatches++;
                    break;
                }
                case CMD_BLIT_COPY_BUFFER: {
                    if (i + 10 > n) { err_code = -17; break; }
                    uint64_t src_h = read_u64_words(words, i);
                    uint64_t src_off = read_u64_words(words, i);
                    uint64_t dst_h = read_u64_words(words, i);
                    uint64_t dst_off = read_u64_words(words, i);
                    uint64_t len = read_u64_words(words, i);
                    id<MTLBuffer> src = get_buf(src_h);
                    id<MTLBuffer> dst = get_buf(dst_h);
                    if (!src || !dst) { err_code = -17; break; }
                    end_active();
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:src sourceOffset:(NSUInteger)src_off
                                toBuffer:dst destinationOffset:(NSUInteger)dst_off
                                    size:(NSUInteger)len];
                    [blit endEncoding];
                    break;
                }
                case CMD_BLIT_FILL_BUFFER: {
                    if (i + 7 > n) { err_code = -18; break; }
                    uint64_t dst_h = read_u64_words(words, i);
                    uint64_t off = read_u64_words(words, i);
                    uint64_t len = read_u64_words(words, i);
                    uint8_t val = (uint8_t)(words[i++] & 0xFFu);
                    id<MTLBuffer> dst = get_buf(dst_h);
                    if (!dst) { err_code = -18; break; }
                    end_active();
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit fillBuffer:dst range:NSMakeRange((NSUInteger)off, (NSUInteger)len) value:val];
                    [blit endEncoding];
                    break;
                }
                case CMD_RENDER_BEGIN: {
                    if (i + 11 > n) { err_code = -19; break; }
                    uint64_t color_h = read_u64_words(words, i);
                    uint32_t load_act = words[i++];
                    float cr = read_f32_word(words, i);
                    float cg = read_f32_word(words, i);
                    float cb_v = read_f32_word(words, i);
                    float ca = read_f32_word(words, i);
                    uint64_t depth_h = read_u64_words(words, i);
                    uint32_t depth_load = words[i++];
                    float clear_d = read_f32_word(words, i);
                    id<MTLTexture> ctex = get_tex(color_h);
                    if (!ctex) { err_code = -20; break; }
                    end_active();
                    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
                    rpd.colorAttachments[0].texture = ctex;
                    rpd.colorAttachments[0].loadAction =
                        (load_act == 2) ? MTLLoadActionClear
                                        : ((load_act == 1) ? MTLLoadActionLoad : MTLLoadActionDontCare);
                    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
                    rpd.colorAttachments[0].clearColor = MTLClearColorMake(cr, cg, cb_v, ca);
                    if (depth_h != 0) {
                        id<MTLTexture> dtex = get_tex(depth_h);
                        if (dtex) {
                            rpd.depthAttachment.texture = dtex;
                            rpd.depthAttachment.loadAction =
                                (depth_load == 2) ? MTLLoadActionClear
                                                  : ((depth_load == 1) ? MTLLoadActionLoad : MTLLoadActionDontCare);
                            rpd.depthAttachment.storeAction = MTLStoreActionStore;
                            rpd.depthAttachment.clearDepth = clear_d;
                        }
                    }
                    renc = [cb renderCommandEncoderWithDescriptor:rpd];
                    break;
                }
                case CMD_RENDER_END: {
                    if (renc) {
                        [renc endEncoding];
                        renc = nil;
                    }
                    break;
                }
                case CMD_SET_RENDER_PSO: {
                    if (i + 2 > n) { err_code = -21; break; }
                    uint64_t h = read_u64_words(words, i);
                    auto it = s.render_psos.find(h);
                    if (it == s.render_psos.end() || !renc) { err_code = -21; break; }
                    [renc setRenderPipelineState:it->second.pso];
                    break;
                }
                case CMD_SET_VIEWPORT: {
                    if (i + 6 > n || !renc) { err_code = -22; break; }
                    float x = read_f32_word(words, i);
                    float y = read_f32_word(words, i);
                    float w = read_f32_word(words, i);
                    float h = read_f32_word(words, i);
                    float zmin = read_f32_word(words, i);
                    float zmax = read_f32_word(words, i);
                    MTLViewport vp = {x, y, w, h, zmin, zmax};
                    [renc setViewport:vp];
                    break;
                }
                case CMD_SET_VERTEX_BUFFER: {
                    if (i + 5 > n || !renc) { err_code = -23; break; }
                    uint32_t slot = words[i++];
                    uint64_t bh = read_u64_words(words, i);
                    uint64_t off = read_u64_words(words, i);
                    id<MTLBuffer> buf = get_buf(bh);
                    if (!buf) { err_code = -23; break; }
                    [renc setVertexBuffer:buf offset:(NSUInteger)off atIndex:slot];
                    break;
                }
                case CMD_SET_VERTEX_BYTES: {
                    if (i + 2 > n || !renc) { err_code = -24; break; }
                    uint32_t slot = words[i++];
                    uint32_t byte_len = words[i++];
                    size_t wlen = (byte_len + 3) / 4;
                    if (i + wlen > n) { err_code = -24; break; }
                    [renc setVertexBytes:&words[i] length:byte_len atIndex:slot];
                    i += wlen;
                    break;
                }
                case CMD_SET_FRAGMENT_BUFFER: {
                    if (i + 5 > n || !renc) { err_code = -25; break; }
                    uint32_t slot = words[i++];
                    uint64_t bh = read_u64_words(words, i);
                    uint64_t off = read_u64_words(words, i);
                    id<MTLBuffer> buf = get_buf(bh);
                    if (!buf) { err_code = -25; break; }
                    [renc setFragmentBuffer:buf offset:(NSUInteger)off atIndex:slot];
                    break;
                }
                case CMD_SET_FRAGMENT_BYTES: {
                    if (i + 2 > n || !renc) { err_code = -26; break; }
                    uint32_t slot = words[i++];
                    uint32_t byte_len = words[i++];
                    size_t wlen = (byte_len + 3) / 4;
                    if (i + wlen > n) { err_code = -26; break; }
                    [renc setFragmentBytes:&words[i] length:byte_len atIndex:slot];
                    i += wlen;
                    break;
                }
                case CMD_SET_FRAGMENT_TEXTURE: {
                    if (i + 3 > n || !renc) { err_code = -27; break; }
                    uint32_t slot = words[i++];
                    uint64_t th = read_u64_words(words, i);
                    id<MTLTexture> tex = get_tex(th);
                    if (!tex) { err_code = -27; break; }
                    [renc setFragmentTexture:tex atIndex:slot];
                    break;
                }
                case CMD_DRAW_PRIMITIVES: {
                    if (i + 4 > n || !renc) { err_code = -28; break; }
                    uint32_t prim = words[i++];
                    uint32_t vstart = words[i++];
                    uint32_t vcount = words[i++];
                    uint32_t icount = words[i++];
                    [renc drawPrimitives:(MTLPrimitiveType)prim
                             vertexStart:vstart
                             vertexCount:vcount
                           instanceCount:icount ? icount : 1];
                    draws++;
                    break;
                }
                case CMD_DRAW_INDEXED: {
                    if (i + 8 > n || !renc) { err_code = -29; break; }
                    uint32_t prim = words[i++];
                    uint32_t idx_count = words[i++];
                    uint32_t idx_type = words[i++];
                    uint64_t ib_h = read_u64_words(words, i);
                    uint64_t ib_off = read_u64_words(words, i);
                    uint32_t icount = words[i++];
                    id<MTLBuffer> ib = get_buf(ib_h);
                    if (!ib) { err_code = -29; break; }
                    [renc drawIndexedPrimitives:(MTLPrimitiveType)prim
                                     indexCount:idx_count
                                      indexType:(MTLIndexType)idx_type
                                    indexBuffer:ib
                              indexBufferOffset:(NSUInteger)ib_off
                                  instanceCount:icount ? icount : 1];
                    draws++;
                    break;
                }
                case CMD_MEMORY_BARRIER: {
                    if (i + 1 > n) { err_code = -30; break; }
                    uint32_t scope = words[i++];
                    if (cenc) {
                        [cenc memoryBarrierWithScope:(MTLBarrierScope)(scope ? scope : MTLBarrierScopeBuffers)];
                    } else if (renc) {
                        [renc memoryBarrierWithScope:(MTLBarrierScope)(scope ? scope : MTLBarrierScopeBuffers)
                                         afterStages:MTLRenderStageFragment
                                        beforeStages:MTLRenderStageVertex];
                    }
                    break;
                }
                default:
                    err_code = -99;
                    break;
            }
        }
        end_active();
        uint64_t t_enc1 = now_ns();
        a->encode_ns = t_enc1 - t_wall0;
        a->draws_encoded = draws;
        a->dispatches_encoded = dispatches;

        if (err_code != 0) {
            a->status = err_code;
            return VM_STATUS_SUCCESS;
        }

        uint64_t next_val = ++s.timeline_value;
        if (s.timeline_event) {
            [cb encodeSignalEvent:s.timeline_event value:next_val];
        }
        [cb commit];
        a->fence_value = next_val;

        if ((a->flags & 1u) != 0) {
            [cb waitUntilCompleted];
            uint64_t t_wall1 = now_ns();
            a->wall_ns = t_wall1 - t_wall0;
            double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
            a->gpu_ns = (gpu_s > 0.0) ? (uint64_t)(gpu_s * 1e9) : a->wall_ns;
            a->status = (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -40;
        } else {
            a->wall_ns = now_ns() - t_wall0;
            a->gpu_ns = 0;
            a->status = 0;
        }
    }
    return VM_STATUS_SUCCESS;
}

static NTSTATUS op_wait_fence(void *raw) {
    auto *a = static_cast<vm_wait_fence_args *>(raw);
    if (!a) return VM_STATUS_INVALID_PARAM;
    auto &s = state();
    id<MTLSharedEvent> ev = nil;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        ev = s.timeline_event;
    }
    if (!ev) {
        a->signaled = 1;
        a->status = 0;
        return VM_STATUS_SUCCESS;
    }
    uint64_t timeout_ms = a->timeout_ns ? std::max<uint64_t>(1, a->timeout_ns / 1000000ULL) : 10000ULL;
    BOOL ok = [ev waitUntilSignaledValue:a->fence_value timeoutMS:timeout_ms];
    a->signaled = ok ? 1u : 0u;
    a->status = 0;
    return VM_STATUS_SUCCESS;
}

extern "C" {
__attribute__((visibility("default"))) extern const unixlib_entry_t __wine_unix_call_funcs[];
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_funcs[] = {
    op_ping,                    // 0
    op_register_buffer,         // 1
    op_unregister_buffer,       // 2
    op_flush_residency,         // 3
    op_bench_zero_copy,         // 4
    op_convert_dxil,            // 5
    op_stats,                   // 6
    op_create_texture,          // 7
    op_destroy_texture,         // 8
    op_upload_texture,          // 9
    op_readback_texture,        // 10
    op_scaler_create,           // 11
    op_scaler_encode,           // 12
    op_scaler_destroy,          // 13
    op_interp_create,           // 14
    op_interp_encode,           // 15
    op_interp_destroy,          // 16
    op_surface_attach,          // 17
    op_surface_present,         // 18
    op_surface_destroy,         // 19
    op_archive_open,            // 20
    op_archive_save,            // 21
    op_archive_close,           // 22
    op_pipeline_create_compute, // 23
    op_pipeline_create_render,  // 24
    op_pipeline_destroy,        // 25
    op_submit_commands,         // 26
    op_wait_fence,              // 27
};

__attribute__((visibility("default"))) extern const unixlib_entry_t __wine_unix_call_wow64_funcs[];
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_wow64_funcs[] = {
    nullptr,
};
}
UNIX_MM

/usr/bin/clang++ -arch x86_64 -ObjC++ -std=c++17 -fobjc-arc -O2 -dynamiclib \
    -mmacosx-version-min=15.0 \
    -framework AppKit -framework Metal -framework MetalFX -framework Foundation -framework QuartzCore \
    -o "$out_dir/x86_64-unix/victory_metal.so" \
    "$tmp/victory_metal_unix.mm"

echo "built $out_dir/x86_64-windows/victory_metal.dll"
echo "built $out_dir/x86_64-unix/victory_metal.so"
