read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_tech.lef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_stdcell.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/dpl/test/aes-opt.def"
set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-metal3
global_route -verbose -allow_congestion

set block [ord::get_db_block]
set gcell_grid [$block getGCellGrid]
set db [ord::get_db]
set tech [$db getTech]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]
set dbu [$block getDefUnits]
puts "ACCURATE_GT_START"
puts "GRID $nx $ny"
puts "DBU $dbu"
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"
for {set li 1} {$li <= 10} {incr li} {
    catch {
        set layer [$tech findRoutingLayer $li]
        if {$layer != "NULL"} {
            for {set gy 0} {$gy < $ny} {incr gy} {
                for {set gx 0} {$gx < $nx} {incr gx} {
                    set cap [$gcell_grid getCapacity $layer $gx $gy]
                    set usg [$gcell_grid getUsage $layer $gx $gy]
                    if {$cap > 0 || $usg > 0} {
                        puts "GC $gx $gy $cap $usg"
                    }
                }
            }
        }
    }
}
puts "ACCURATE_GT_END"
