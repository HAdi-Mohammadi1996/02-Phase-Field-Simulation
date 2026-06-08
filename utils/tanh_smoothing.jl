using ImageFiltering

function kernel_tuple(K1::AbstractVector, N::Int)
    len = length(K1)
    tuple((
        centered(reshape(K1, ntuple(d -> d == i ? len : 1, N)...))
        for i in 1:N
    )...)
end

function tanh_ND(f::AbstractArray, zeta::Real)
    # --- Build 1D tanh kernel ---
    half_width_factor = 5
    Nw = ceil(Int, half_width_factor * zeta)
    x = -Nw:Nw
    K1 = (1/(4*zeta)) .* (1 ./ cosh.(x ./ (2*zeta))).^2
    K1 ./= sum(K1)

    # --- Wrap kernel for convolution (CenteredKernel is the right one) ---
    kernel = kernel_tuple(K1, ndims(f))

    fs = imfilter(f, kernel,"replicate")
    return fs

end