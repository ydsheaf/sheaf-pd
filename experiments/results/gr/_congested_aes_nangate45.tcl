read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def"

set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-metal3
global_route -verbose -allow_congestion

set block [ord::get_db_block]
set die [$block getDieArea]
set dbu [$block getDefUnits]
puts "DIE_UM [expr [$die xMin] / double($dbu)] [expr [$die yMin] / double($dbu)] [expr [$die xMax] / double($dbu)] [expr [$die yMax] / double($dbu)]"
puts "DBU $dbu"

# Output net bounding boxes (fast for small designs)
puts "=== NET_BBOX_START ==="
set nets [$block getNets]
foreach net $nets {
    set iters [$net getITerms]
    if { [llength $iters] < 2 } continue
    set xmin 1e18; set ymin 1e18; set xmax -1e18; set ymax -1e18
    foreach it $iters {
        set inst [$it getInst]
        if { $inst == "NULL" } continue
        set bb [$inst getBBox]
        set cx [expr ([$bb xMin] + [$bb xMax]) / 2.0 / $dbu]
        set cy [expr ([$bb yMin] + [$bb yMax]) / 2.0 / $dbu]
        if { $cx < $xmin } { set xmin $cx }
        if { $cy < $ymin } { set ymin $cy }
        if { $cx > $xmax } { set xmax $cx }
        if { $cy > $ymax } { set ymax $cy }
    }
    if { $xmin < 1e17 } {
        puts "NB $xmin $ymin $xmax $ymax"
    }
}
puts "=== NET_BBOX_END ==="
