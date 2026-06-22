using JLD2
using MAT
using Printf

# ==============================================================================
# USER SETTINGS - change only these three paths, then run this file.
# ==============================================================================

ORIGINAL_MAT_PATH = raw"C:\Users\r43341mm\OneDrive - The University of Manchester\Research\SharedData\PhaseFieldResults\Validation3D\W0\mat/t000000.mat"
CHECKPOINT_DIR = raw"C:\Users\r43341mm\OneDrive - The University of Manchester\Research\SharedData\PhaseFieldResults\Validation3D\W0\results"
OUTPUT_DIR = raw"C:\Users\r43341mm\OneDrive - The University of Manchester\Research\SharedData\PhaseFieldResults\Validation3D\W0\mat_volume_preserving"

# ==============================================================================

const ORIGINAL_PHASE_KEY = "C"
const CHECKPOINT_FIELD_KEY = "phi"
const OUTPUT_PHASE_KEY = "C"

"""
Reconstruct a ternary volume while preserving the original number of Ni voxels.

YSZ is fixed from the original segmented volume. Among all non-YSZ voxels, the
`ni_voxel_count` largest values of `phi` are labelled Ni and the rest are pore.
The tie-breaking order is deterministic (Julia linear-index order).
"""
function reconstruct_volume_preserving(phi, ysz_mask, ni_voxel_count)
    size(phi) == size(ysz_mask) ||
        throw(DimensionMismatch("phi and the original YSZ mask have different sizes"))
    all(isfinite, phi) ||
        throw(ArgumentError("phi contains NaN or Inf values"))

    available_voxels = count(!, ysz_mask)
    0 <= ni_voxel_count <= available_voxels ||
        throw(ArgumentError("requested Ni count is outside the non-YSZ domain"))

    phase_volume = fill(Int32(3), size(phi))
    phase_volume[ysz_mask] .= Int32(2)

    ni_voxel_count == 0 && return phase_volume, Inf

    if ni_voxel_count == available_voxels
        phase_volume[.!ysz_mask] .= Int32(1)
        return phase_volume, -Inf
    end

    # Find the value of the Ni/pore cutoff without fully sorting the field.
    non_ysz_values = Vector{eltype(phi)}(undef, available_voxels)
    position = 0
    @inbounds for index in eachindex(phi, ysz_mask)
        if !ysz_mask[index]
            position += 1
            non_ysz_values[position] = phi[index]
        end
    end
    cutoff = partialsort!(non_ysz_values, ni_voxel_count; rev=true)

    # Values strictly above the cutoff are unambiguous Ni.
    selected = 0
    @inbounds for index in eachindex(phi, ysz_mask)
        if !ysz_mask[index] && phi[index] > cutoff
            phase_volume[index] = Int32(1)
            selected += 1
        end
    end

    # If several voxels equal the cutoff, select only as many as required.
    remaining = ni_voxel_count - selected
    @inbounds for index in eachindex(phi, ysz_mask)
        remaining == 0 && break
        if !ysz_mask[index] && phi[index] == cutoff
            phase_volume[index] = Int32(1)
            remaining -= 1
        end
    end

    remaining == 0 ||
        error("failed to select the requested number of Ni voxels")

    return phase_volume, cutoff
end

function main()
    original_mat_path = abspath(ORIGINAL_MAT_PATH)
    checkpoint_dir = abspath(CHECKPOINT_DIR)
    output_dir = abspath(OUTPUT_DIR)

    println("Starting volume-preserving reconstruction...")
    println("Original data: ", original_mat_path)
    println("Checkpoints:   ", checkpoint_dir)
    println("Output:        ", output_dir)
    flush(stdout)

    isfile(original_mat_path) ||
        error("original .mat file does not exist: $original_mat_path")
    isdir(checkpoint_dir) ||
        error("checkpoint folder does not exist: $checkpoint_dir")
    mkpath(output_dir)

    original_data = matread(original_mat_path)
    haskey(original_data, ORIGINAL_PHASE_KEY) ||
        error("original .mat file does not contain variable '$ORIGINAL_PHASE_KEY'")

    original_phase = original_data[ORIGINAL_PHASE_KEY]
    ysz_mask = original_phase .== 2
    ni_voxel_count = count(==(1), original_phase)

    checkpoint_files = sort(filter(
        path -> endswith(lowercase(path), ".jld2"),
        readdir(checkpoint_dir; join=true),
    ))
    isempty(checkpoint_files) &&
        error("no .jld2 checkpoint files found in: $checkpoint_dir")

    @printf(
        "Original volume: Ni=%d, YSZ=%d, pore=%d, total=%d\n",
        ni_voxel_count,
        count(ysz_mask),
        length(original_phase) - ni_voxel_count - count(ysz_mask),
        length(original_phase),
    )

    for (file_number, checkpoint_path) in enumerate(checkpoint_files)
        phi = JLD2.load(checkpoint_path, CHECKPOINT_FIELD_KEY)
        phase_volume, cutoff =
            reconstruct_volume_preserving(phi, ysz_mask, ni_voxel_count)

        output_name = splitext(basename(checkpoint_path))[1] * ".mat"
        output_path = joinpath(output_dir, output_name)
        matopen(output_path, "w") do file
            write(file, OUTPUT_PHASE_KEY, phase_volume)
        end

        @printf(
            "[%d/%d] %s -> %s (adaptive cutoff = %.10g)\n",
            file_number,
            length(checkpoint_files),
            basename(checkpoint_path),
            output_name,
            cutoff,
        )
    end

    println("Finished. Reconstructed files were saved in:")
    println(output_dir)
end

# Run immediately when this file is executed from an editor or included in a REPL.
main()
