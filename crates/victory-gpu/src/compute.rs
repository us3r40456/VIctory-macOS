//! Compute pipelines built from HLSL at run time.
//!
//! The root signature is serialized here from [`Param`]s instead of
//! being embedded with `[RootSignature]`: Wine's HLSL compiler
//! (vkd3d-shader) ignores that attribute ("W5302: Ignoring unknown
//! attribute") and writes no root signature part, so under Wine creating a
//! root signature from the shader's bytecode fails with `E_FAIL`. The
//! signature is version 1.1 with no flags, as Microsoft's compilers build
//! it from an HLSL root signature string.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Error, PCSTR, Result, s};

/// A root parameter, visible to every stage.
pub(crate) enum Param<'a> {
    /// A descriptor table of ranges, one after another.
    Table(&'a [Range]),
    /// `count` 32-bit constants in `b{register}`.
    Constants { register: u32, count: u32 },
    /// A root shader resource view in `t{register}, space{space}`.
    Srv { register: u32, space: u32 },
    /// A root unordered access view in `u{register}`.
    Uav { register: u32 },
}

/// A descriptor table's range in space 0.
pub(crate) struct Range {
    kind: D3D12_DESCRIPTOR_RANGE_TYPE,
    register: u32,
    count: u32,
}

/// `count` shader resource views from `t{register}`.
pub(crate) const fn srv(register: u32, count: u32) -> Range {
    Range {
        kind: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        register,
        count,
    }
}

/// `count` unordered access views from `u{register}`.
pub(crate) const fn uav(register: u32, count: u32) -> Range {
    Range {
        kind: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        register,
        count,
    }
}

/// Compiles `source`'s `main` for `cs_5_1` with the compiler `flags`, and
/// makes its pipeline with a root signature of `params`. The compiler's
/// messages for a failed compile are logged; Wine's compiler answers
/// `E_NOTIMPL` for features it does not have.
pub(crate) fn pipeline(
    source: &str,
    name: PCSTR,
    flags: u32,
    params: &[Param],
) -> Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let device = &crate::host::host().ok_or_else(Error::empty)?.device;
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            flags,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes, and a name the caller keeps.
            unsafe {
                let text = std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                );
                tracing::warn!("{}: {}", name.display(), String::from_utf8_lossy(text));
            }
        }
        return Err(e);
    }
    let code = code.ok_or_else(Error::empty)?;
    let root = root_signature(device, params)?;
    // SAFETY: the bytecode and the root signature live through the call.
    let pso = unsafe {
        device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.GetBufferPointer(),
                BytecodeLength: code.GetBufferSize(),
            },
            ..Default::default()
        })
    }?;
    // A pass of ours, by the name it was compiled under: what a frame
    // replay's reports call it (`VICTORY_RECORD`).
    // SAFETY: the caller's name, a string that ends with a NUL.
    crate::host::name(&pso, || format!("victory: {}", unsafe { name.display() }));
    Ok((root, pso))
}

pub(crate) fn root_signature(device: &ID3D12Device, params: &[Param]) -> Result<ID3D12RootSignature> {
    let ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE1>> = params
        .iter()
        .map(|p| match p {
            Param::Table(ranges) => ranges
                .iter()
                .map(|r| D3D12_DESCRIPTOR_RANGE1 {
                    RangeType: r.kind,
                    NumDescriptors: r.count,
                    BaseShaderRegister: r.register,
                    RegisterSpace: 0,
                    Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
                    OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    let descriptor = |kind, register, space| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: register,
                RegisterSpace: space,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params: Vec<D3D12_ROOT_PARAMETER1> = params
        .iter()
        .zip(&ranges)
        .map(|(p, ranges)| match *p {
            Param::Table(_) => D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            Param::Constants { register, count } => D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: register,
                        RegisterSpace: 0,
                        Num32BitValues: count,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            Param::Srv { register, space } => {
                descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, register, space)
            }
            Param::Uav { register } => descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV, register, 0),
        })
        .collect();
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        },
    };
    let (mut blob, mut error): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}
