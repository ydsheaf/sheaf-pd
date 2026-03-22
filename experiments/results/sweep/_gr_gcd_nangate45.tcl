read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def"

set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-metal3
global_route -verbose -allow_congestion


set block [ord::get_db_block]
set dbu [$block getDefUnits]
set die [$block getDieArea]
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]

puts "GRID $nx $ny"
puts "DBU $dbu"
puts "DIE [$die xMin] [$die yMin] [$die xMax] [$die yMax]"

# Write segments for guide counting
set seg_file "/mnt/storage1/users/ydwu/claude_projects/sheaf-pd/experiments/results/sweep/seg_gcd_nangate45.txt"
grt::write_segments $seg_file

# Output grid coordinates
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"
