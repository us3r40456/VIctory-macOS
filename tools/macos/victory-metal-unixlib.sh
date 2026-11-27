#!/bin/sh
# Builds the Route A.5 Wine unixlib bridge (victory_metal.dll + victory_metal.so)
# for zero-copy Apple Silicon unified memory, global MTLResidencySet tracking,
# and in-process DXIL -> MetalLib conversion via libmetalirconverter.dylib.
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
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <mutex>
#include <unordered_map>
#include <vector>

typedef int32_t NTSTATUS;
typedef NTSTATUS (*unixlib_entry_t)(void *args);

static constexpr NTSTATUS VM_STATUS_SUCCESS = 0;
static constexpr NTSTATUS VM_STATUS_INVALID_PARAM = (NTSTATUS)0xC000000Du;
static constexpr NTSTATUS VM_STATUS_UNSUCCESSFUL = (NTSTATUS)0xC0000001u;
static constexpr uint32_t VM_ABI_VERSION = 1;

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

static uint64_t now_ns() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (mach_absolute_time() * tb.numer) / tb.denom;
}

static const char *BENCH_MSL = R"MSL(
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
)MSL";

struct MetalState {
    std::mutex mu;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    id<MTLComputePipelineState> bench_sum_pso = nil;
    id<MTLComputePipelineState> bench_write_pso = nil;
    id<MTLBuffer> bench_partials = nil;

    uint64_t next_handle = 1;
    uint32_t pending_residency = 0;
    bool has_residency = false;
    bool has_metalfx = false;
    uint32_t cached_flags = 0;
    uint64_t cached_vram = 0;
    uint64_t cached_max_buf = 0;
    char cached_name[128] = {0};

    std::unordered_map<uint64_t, BufferEntry> by_handle;
    std::unordered_map<uint64_t, uint64_t> backing_to_handle;

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
            uint32_t f = 0;
            if (device.hasUnifiedMemory) f |= 1u << 0;
            if (has_residency) f |= 1u << 1;
            if (has_metalfx) f |= 1u << 2;
            if (ir::api().loaded()) f |= 1u << 3;
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

    bool init_bench_locked() {
        if (bench_sum_pso && bench_write_pso && bench_partials) return true;
        @autoreleasepool {
            NSError *err = nil;
            NSString *src = [NSString stringWithUTF8String:BENCH_MSL];
            id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
            if (!lib) return false;
            id<MTLFunction> f_sum = [lib newFunctionWithName:@"bench_sum"];
            id<MTLFunction> f_write = [lib newFunctionWithName:@"bench_write"];
            if (!f_sum || !f_write) return false;
            bench_sum_pso = [device newComputePipelineStateWithFunction:f_sum error:&err];
            bench_write_pso = [device newComputePipelineStateWithFunction:f_write error:&err];
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
        return bench_sum_pso && bench_write_pso && bench_partials;
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
        if (!s.init_locked() || !s.init_bench_locked()) return VM_STATUS_UNSUCCESSFUL;
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

extern "C" {
__attribute__((visibility("default"))) extern const unixlib_entry_t __wine_unix_call_funcs[];
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_funcs[] = {
    op_ping,
    op_register_buffer,
    op_unregister_buffer,
    op_flush_residency,
    op_bench_zero_copy,
    op_convert_dxil,
    op_stats,
};

__attribute__((visibility("default"))) extern const unixlib_entry_t __wine_unix_call_wow64_funcs[];
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_wow64_funcs[] = {
    nullptr,
};
}
UNIX_MM

/usr/bin/clang++ -arch x86_64 -ObjC++ -std=c++17 -fobjc-arc -O2 -dynamiclib \
    -mmacosx-version-min=15.0 \
    -framework Metal -framework MetalFX -framework Foundation -framework QuartzCore \
    -o "$out_dir/x86_64-unix/victory_metal.so" \
    "$tmp/victory_metal_unix.mm"

echo "built $out_dir/x86_64-windows/victory_metal.dll"
echo "built $out_dir/x86_64-unix/victory_metal.so"
