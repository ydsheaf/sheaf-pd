#!/bin/bash
# Run OpenROAD GR on all designs and collect congestion reports
set -e

OPENROAD=/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
ORFS=/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts
RESULTS=experiments/results/gr
mkdir -p $RESULTS

run_gr() {
    local name=$1
    local tcl_file="$RESULTS/_gr_${name}.tcl"
    local log_file="$RESULTS/openroad_${name}.log"
    local rpt_file="$RESULTS/congestion_${name}.rpt"

    echo "=== Running GR: $name ==="

    # Write Tcl script
    cat > "$tcl_file" << TCLEOF
puts "=== OpenROAD GR: $name ==="
$2
$3

set_global_routing_layer_adjustment metal1-metal10 0.5
global_route -verbose -congestion_report_file "$rpt_file"

puts "=== LAYER STATISTICS ==="
report_routing_layer_statistics

puts "=== DONE: $name ==="
TCLEOF

    $OPENROAD -no_splash -exit "$tcl_file" > "$log_file" 2>&1
    echo "  Log: $log_file"
    echo "  Report: $rpt_file"
    tail -20 "$log_file"
    echo
}

# gcd_nangate45
run_gr "gcd_nangate45" \
    "read_lef $ORFS/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef
read_lef $ORFS/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef" \
    "read_def $ORFS/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def"

# aes_nangate45
run_gr "aes_nangate45" \
    "read_lef $ORFS/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef
read_lef $ORFS/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef" \
    "read_def $ORFS/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def"

# gcd_sky130
run_gr "gcd_sky130" \
    "read_lef $ORFS/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef
read_lef $ORFS/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef" \
    "read_def $ORFS/flow/tutorials/scripts/drt/gcd/4_cts.def"

echo "=== All GR runs complete ==="
