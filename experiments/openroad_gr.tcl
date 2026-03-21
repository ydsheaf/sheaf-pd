# OpenROAD Global Routing — extract per-G-cell congestion
# Usage: openroad -exit experiments/openroad_gr.tcl <design_name>
#
# Reads LEF/DEF, runs global routing, reports congestion stats.
# Output goes to stdout — parsed by validate_eta_drc.py.

set design_name [lindex $argv 0]
set orfs "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"

puts "=== OpenROAD GR: $design_name ==="

switch $design_name {
    "gcd_nangate45" {
        read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
        read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
        read_def "$orfs/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def"
    }
    "aes_nangate45" {
        read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
        read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
        read_def "$orfs/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def"
    }
    "gcd_sky130" {
        read_lef "$orfs/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef"
        read_lef "$orfs/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"
        read_def "$orfs/flow/tutorials/scripts/drt/gcd/4_cts.def"
    }
    default {
        puts "ERROR: Unknown design $design_name"
        exit 1
    }
}

# Run global routing
set_global_routing_layer_adjustment metal1-metal10 0.5
global_route -verbose -congestion_report_file "experiments/results/gr/congestion_${design_name}.rpt"

# Report layer statistics
puts "=== LAYER STATISTICS ==="
report_routing_layer_statistics

# Report congestion in grid format
puts "=== CONGESTION GRID ==="
puts "Design: $design_name"

# Get die area
set die_area [ord::get_db_block]
set die_box [$die_area getDieArea]
puts "Die: [$die_box xMin] [$die_box yMin] [$die_box xMax] [$die_box yMax]"

puts "=== DONE ==="
