# Convert JLD2 simulation checkpoints to 3-phase .mat volumes.

function reconstruct_ternary(phi, ysz_mask, threshold=0.5)
    # I have checked there is no need for clamping here
    # becasue we are only intersted in whether phi is >= 0.5 or not
    # threshold comparison is unaffected by out-of-[0,1] overshoot
    
    phase_vol = fill(Int32(3), size(phi))
    phase_vol[phi .>= threshold] .= Int32(1)
    phase_vol[ysz_mask] .= Int32(2)
    return phase_vol
end

