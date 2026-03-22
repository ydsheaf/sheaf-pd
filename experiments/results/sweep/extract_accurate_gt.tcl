# extract_accurate_gt.tcl — Per-G-cell capacity/usage from dbGCellGrid (Issue #23)
#
# Usage:
#   openroad -no_splash -exit extract_accurate_gt.tcl
#
# Environment variables (set defaults for gcd_nangate45):
#   LEF_FILES    — space-separated LEF paths
#   DEF_FILE     — DEF path
#   ADJUST_LAYERS — layer adjustment range (e.g. metal1-metal10)
#   ADJUST_VALUE  — adjustment fraction (e.g. 0.8)
#   SIGNAL_LAYERS — routing layer range (e.g. metal2-metal3)

# Defaults for gcd_nangate45
if {![info exists env(LEF_FILES)]} {
    set lef_files [list \
        "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef" \
        "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"]
} else {
    set lef_files $env(LEF_FILES)
}

if {![info exists env(DEF_FILE)]} {
    set def_file "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def"
} else {
    set def_file $env(DEF_FILE)
}

if {![info exists env(ADJUST_LAYERS)]} {
    set adjust_layers "metal1-metal10"
} else {
    set adjust_layers $env(ADJUST_LAYERS)
}

if {![info exists env(ADJUST_VALUE)]} {
    set adjust_value 0.8
} else {
    set adjust_value $env(ADJUST_VALUE)
}

if {![info exists env(SIGNAL_LAYERS)]} {
    set signal_layers "metal2-metal3"
} else {
    set signal_layers $env(SIGNAL_LAYERS)
}

# Step 1: Load design
foreach lef $lef_files {
    read_lef $lef
}
read_def $def_file

# Step 2: Run congested GR
set_global_routing_layer_adjustment $adjust_layers $adjust_value
set_routing_layers -signal $signal_layers
global_route -verbose -allow_congestion

# Step 3: Read per-G-cell capacity and usage from dbGCellGrid
set block [ord::get_db_block]
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]
set dbu [$block getDefUnits]

# Get tech for layer lookup
set db [ord::get_db]
set tech [$db getTech]

puts "ACCURATE_GT_START"
puts "GRID $nx $ny"
puts "DBU $dbu"

# Output grid coordinates for coordinate mapping
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"

# Collect all routing layers from tech
set routing_layers [list]
foreach layer [$tech getLayers] {
    if {[$layer getType] == "ROUTING"} {
        lappend routing_layers $layer
    }
}
puts "NLAYERS [llength $routing_layers]"

# For each G-cell, sum capacity and usage across all routing layers
for {set gy 0} {$gy < $ny} {incr gy} {
    for {set gx 0} {$gx < $nx} {incr gx} {
        set total_cap 0.0
        set total_usage 0.0
        foreach layer $routing_layers {
            catch {
                set cap [$gcell_grid getCapacity $layer $gx $gy]
                set usg [$gcell_grid getUsage $layer $gx $gy]
                set total_cap [expr {$total_cap + $cap}]
                set total_usage [expr {$total_usage + $usg}]
            }
        }
        if {$total_cap > 0 || $total_usage > 0} {
            puts "GC $gx $gy $total_cap $total_usage"
        }
    }
}
puts "ACCURATE_GT_END"
