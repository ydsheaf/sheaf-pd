# extract_accurate_gt_aes_psm_nangate45.tcl
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/psm/test/Nangate45_data/aes.def"
set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-metal3
global_route -verbose -allow_congestion

set block [ord::get_db_block]
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]
set dbu [$block getDefUnits]
set db [ord::get_db]
set tech [$db getTech]

puts "ACCURATE_GT_START"
puts "GRID $nx $ny"
puts "DBU $dbu"
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"

set routing_layers [list]
foreach layer [$tech getLayers] {
    if {[$layer getType] == "ROUTING"} {
        lappend routing_layers $layer
    }
}
puts "NLAYERS [llength $routing_layers]"

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
