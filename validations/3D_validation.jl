import DifferentialEquations as DE
include("../utils/tanh_smoothing.jl")
using Plots
using LoopVectorization
using Base.Threads
import BenchmarkTools as BT
include("../utils/reconstruct.jl")
using MAT
using JLD2
import Sundials
using ProgressLogging

output_dir = raw"C:\Users\r43341mm\OneDrive - The University of Manchester\Research\SharedData\PhaseFieldResults/"
jld_dir = joinpath(output_dir, "Validation3D/W+0.5/results")
mat_dir = joinpath(output_dir, "Validation3D/W+0.5/mat")
gif_dir = joinpath(output_dir, "gifs/Validation")
mkpath(jld_dir)
mkpath(mat_dir)
mkpath(gif_dir)


const Lx = 100.0
const Ly = 100.0
const Lz = 100.0
const Nx = 100
const Ny = 100
const Nz = 100
const dx = Lx / Nx
const dy = Ly / Ny
const dz = Lz / Nz

x = range(start=1, stop=Lx, length=Nx)
y = range(start=1, stop=Ly, length=Ny)
z = range(start=1, stop=Lz, length=Nz)

function norm_calculation3D!(norm2_c, c)

    inv_dx2 = 1.0/(dx*dx)
    inv_dy2 = 1.0/(dy*dy)
    inv_dz2 = 1.0/(dz*dz)

    @inbounds @simd for k in 1:Nz
        @inbounds @simd for j in 1:Ny
            @inbounds @simd for i in 1:Nx

                ce = i== Nx ? c[i, j, k] : c[i + 1, j, k]   # no flux boundary condition
                cw = i == 1 ? c[i, j, k] : c[i - 1, j, k]   # no flux boundary condition
                cn = j== Ny ? c[i, j, k] : c[i, j + 1, k]   # no flux boundary condition
                cs = j == 1 ? c[i, j, k] : c[i, j - 1, k]   # no flux boundary condition
                ct = k== Nz ? c[i, j, k] : c[i, j, k + 1]   # no flux boundary condition
                cb = k == 1 ? c[i, j, k] : c[i, j, k - 1]   # no flux boundary condition

                norm2_c[i, j, k] = 0.25*(ce-cw)^2 * inv_dx2 +
                                   0.25*(cn-cs)^2 * inv_dy2 +
                                   0.25*(ct-cb)^2 * inv_dz2
            end
        end
    end
    return nothing
end

function mobility!(mobil, norm2_c_YSZ, norm2_c_Ni)
    a1, a2, a3, ω = 1.0, 0.1, 0.1, 0.01
    invω = 1.0/ω
    @turbo for i in 1:Nx, j in 1:Ny, k in 1:Nz
        tysz = tanh(norm2_c_YSZ[i,j,k] * invω)
        tni  = tanh(norm2_c_Ni[i,j,k]  * invω)

        mobil[i,j,k] =
            a1 * (1 - tysz) * tni +
            a2 * tysz * tni +
            a3 * tysz * (1 - tni)
    end
    return nothing
end

function lap_c_YSZ!(lap_c, c_YSZ)
    inv_dx2 = 1.0/(dx*dx)
    inv_dy2 = 1.0/(dy*dy)
    inv_dz2 = 1.0/(dz*dz)

    @inbounds @simd for k in 1:Nz
        @inbounds @simd for j in 1:Ny
            @inbounds @simd for i in 1:Nx

                cp = c_YSZ[i, j, k]
                ce = i== Nx ? c_YSZ[i, j, k] : c_YSZ[i + 1, j, k]   # no flux boundary condition
                cw = i == 1 ? c_YSZ[i, j, k] : c_YSZ[i - 1, j, k]   # no flux boundary condition
                cn = j== Ny ? c_YSZ[i, j, k] : c_YSZ[i, j + 1, k]   # no flux boundary condition
                cs = j == 1 ? c_YSZ[i, j, k] : c_YSZ[i, j - 1, k]   # no flux boundary condition
                ct = k== Nz ? c_YSZ[i, j, k] : c_YSZ[i, j, k + 1]   # no flux boundary condition
                cb = k == 1 ? c_YSZ[i, j, k] : c_YSZ[i, j, k - 1]   # no flux boundary condition

                lap_c[i, j, k] = (ce -2*cp + cw) * inv_dx2 + 
                                 (cn -2*cp + cs) * inv_dy2 +
                                 (ct -2*cp + cb) * inv_dz2
            end
        end
    end
    return nothing
end

function main_3D!(du, u, p, t)
    c_YSZ, lap_c, norm2_c_YSZ, mobil, mu = p

    A1 = 1.0
    A12 = 0.7
    κ11 = 1.1
    κ12 = 0.8
    W_ca = 0.5

    inv_dx2 = inv(dx*dx)
    inv_dy2 = inv(dy*dy)
    inv_dz2 = inv(dz*dz)

    norm_calculation3D!(norm2_c_Ni, u)
    mobility!(mobil, norm2_c_YSZ, norm2_c_Ni)

    @inbounds @simd for k in 1:Nz
        @inbounds @simd for j in 1:Ny
            @inbounds @simd for i in 1:Nx

                up = u[i, j, k]
                ue = i== Nx ? u[i, j, k] : u[i + 1, j, k]   # no flux boundary condition
                uw = i == 1 ? u[i, j, k] : u[i - 1, j, k]   # no flux boundary condition
                un = j== Ny ? u[i, j, k] : u[i, j + 1, k]   # no flux boundary condition
                us = j == 1 ? u[i, j, k] : u[i, j - 1, k]   # no flux boundary condition
                ut = k== Nz ? u[i, j, k] : u[i, j, k + 1]   # no flux boundary condition
                ub = k == 1 ? u[i, j, k] : u[i, j, k - 1]   # no flux boundary condition

                cp = c_YSZ[i, j, k]

                dfdc = A1 * 2 * (up*(1-up)^2 - up^2*(1-up)) +
                       A12 * 2 * ((up+cp)*(1-up-cp)^2 - (up+cp)^2*(1-up-cp)) + 
                       W_ca * norm2_c_YSZ[i, j, k]

                lap_u = (ue -2*up + uw) * inv_dx2 + 
                        (un -2*up + us) * inv_dy2 + 
                        (ut -2*up + ub) * inv_dz2

                mu[i, j, k] = dfdc - κ11 * lap_u - 0.5 * κ12 * lap_c[i, j, k]
            end
        end
    end
    @inbounds @simd for k in 1:Nz
        @inbounds @simd for j in 1:Ny
            @inbounds @simd for i in 1:Nx

                inv_Mi = 1.0 / mobil[i, j, k]

                Me = i== Nx ? 0.0 : 2.0 / (1/mobil[i+1, j, k] + inv_Mi)
                Mw = i == 1 ? 0.0 : 2.0 / (1/mobil[i-1, j, k] + inv_Mi)
                Mn = j== Ny ? 0.0 : 2.0 / (1/mobil[i, j+1, k] + inv_Mi)
                Ms = j == 1 ? 0.0 : 2.0 / (1/mobil[i, j-1, k] + inv_Mi)
                Mt = k== Nz ? 0.0 : 2.0 / (1/mobil[i, j, k+1] + inv_Mi)
                Mb = k == 1 ? 0.0 : 2.0 / (1/mobil[i, j, k-1] + inv_Mi)


                mu_p = mu[i, j, k]
                mu_e = i== Nx ? mu[i, j, k] : mu[i + 1, j, k]
                mu_w = i == 1 ? mu[i, j, k] : mu[i - 1, j, k]
                mu_n = j== Ny ? mu[i, j, k] : mu[i, j + 1, k]
                mu_s = j == 1 ? mu[i, j, k] : mu[i, j - 1, k]
                mu_t = k== Nz ? mu[i, j, k] : mu[i, j, k + 1]
                mu_b = k == 1 ? mu[i, j, k] : mu[i, j, k - 1]

                du[i, j, k]= (Me * mu_e - (Me + Mw) * mu_p + Mw * mu_w) * inv_dx2 + 
                             (Mn * mu_n - (Mn + Ms) * mu_p + Ms * mu_s) * inv_dy2 + 
                             (Mt * mu_t - (Mt + Mb) * mu_p + Mb * mu_b) * inv_dz2
            end
        end
    end
end

u0 = zeros(Nx, Ny, Nz)
u0[35:65, 35:65, 35:65] .= 1.0
c_YSZ = zeros(Nx, Ny, Nz)
c_YSZ[:, 1:35, :] .= 1.0
c_YSZ = tanh_ND(c_YSZ, 0.4)
mu = zeros(Nx, Ny, Nz)
mobil = zeros(Nx, Ny, Nz)
norm2_c_YSZ = zeros(Nx, Ny, Nz)
norm2_c_Ni = zeros(Nx, Ny, Nz)
lap_c = zeros(Nx, Ny, Nz)

norm_calculation3D!(norm2_c_YSZ, c_YSZ)
lap_c_YSZ!(lap_c, c_YSZ)
p = (c_YSZ, lap_c, norm2_c_YSZ, mobil, mu)

prob = DE.ODEProblem(main_3D!, u0, (0, 5000.0), p)
si = zeros(Nx, Ny)
@time sol = DE.solve(prob, Sundials.CVODE_BDF(; linear_solver = :GMRES); saveat = 50.0, progress = true);
@. si = sol.u[end][:, :, Int(Nz/2)] + 0.5 * c_YSZ[:, :, Int(Nz/2)] # extract the middle z slice for visualization

anim = @animate for i in 1:length(sol.u)
    @. si = sol.u[i][:, :, Int(Nz/2)] + 0.5 * c_YSZ[:, :, Int(Nz/2)]
    heatmap(x, y, si', xlabel="x", ylabel="y", color= :grays, aspect_ratio=1)
end
gif(anim, joinpath(gif_dir, "3D_W+0.5.gif"), fps = 5)

for (i, phi) in enumerate(sol.u)
    step = i - 1
    step_file = joinpath(jld_dir, "t$(lpad(step, 6, '0')).jld2")
    jldopen(step_file, "w"; compress=ZstdFilter(9)) do f
        f["phi"] = Array(phi)
        f["step"] = step
        f["time"] = sol.t[i]
    end
    mat_path = joinpath(mat_dir, "t$(lpad(step, 6, '0')).mat")
    phase_vol = reconstruct_ternary(Array(phi), (c_YSZ .>0.5))
    matopen(mat_path, "w") do f
        write(f, "C", phase_vol)
    end
end