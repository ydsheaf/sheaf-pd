set orfs "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"

# Use aes which has the most nets
read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
read_lef "$orfs/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
read_def "$orfs/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def"

# Restrict routing to fewer layers to create artificial congestion
set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-metal3

global_route -verbose -allow_congestion

puts "=== CONGESTION REPORT ==="
