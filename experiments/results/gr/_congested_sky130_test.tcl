set orfs "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
read_lef "$orfs/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef"
read_lef "$orfs/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"
read_def "$orfs/flow/tutorials/scripts/drt/gcd/4_cts.def"

set_global_routing_layer_adjustment met1-met5 0.95
set_routing_layers -signal met2-met3
global_route -verbose -allow_congestion

set block [ord::get_db_block]
set dbu [$block getDefUnits]
set die [$block getDieArea]
puts "DIE_UM [expr [$die xMin] / double($dbu)] [expr [$die yMin] / double($dbu)] [expr [$die xMax] / double($dbu)] [expr [$die yMax] / double($dbu)]"
puts "DBU $dbu"

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
