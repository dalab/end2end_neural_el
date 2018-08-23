require 'torch'
require 'nn'
require 'os'
el_path = os.getenv("EL_PATH")
cmd = torch.CmdLine()
cmd:option('-ent_vecs_filepath', '/home/end2end_neural_el/data/entities/ent_vecs/ent_vecs__ep_147.t7', 'the file paht for the entity vectors with best score.')
opt = cmd:parse(arg or {})
folder_name, file_name = string.match(opt.ent_vecs_filepath, "(.-)([^/]*)$")
print("folder_name: " .. folder_name)
print("file_name: " .. file_name)
--print(opt.ent_vecs_filepath)

ent_vecs = torch.load(opt.ent_vecs_filepath)
print('lines = ' .. ent_vecs:size(1))
print('columns = ' .. ent_vecs:size(2))
ent_vecs = nn.Normalize(2):forward(ent_vecs:double()) -- Needs to be normalized to have norm 1.
-- print them to normal txt file
--out = assert(io.open(el_path .. "/data/entities/ent_vecs/ent_vecs.txt", "w")) -- open a file for serialization
out = assert(io.open(folder_name .. "ent_vecs.txt", "w")) -- open a file for serialization
splitter = " "
for i=1,ent_vecs:size(1) do
    for j=1,ent_vecs:size(2) do
        out:write(ent_vecs[i][j])
        if j == ent_vecs:size(2) then
            out:write("\n")
        else
            out:write(splitter)
        end
    end
end
out:close()


