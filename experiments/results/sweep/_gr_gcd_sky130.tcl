read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/flow/tutorials/scripts/drt/gcd/4_cts.def"

set_global_routing_layer_adjustment met1-met5 0.95
set_routing_layers -signal met2-met3
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
set seg_file "/mnt/storage1/users/ydwu/claude_projects/sheaf-pd/experiments/results/sweep/seg_gcd_sky130.txt"
grt::write_segments $seg_file

# Output grid coordinates
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"
