import torch 
path = "/home/kefe11/Results_25mm_128p_norm/UNet3DTo2D/params_34"
model0 = torch.load(f"{path}/model_0.pt", map_location=torch.device('cpu'))
model1 = torch.load(f"{path}/model_1.pt", map_location=torch.device('cpu'))
model2 = torch.load(f"{path}/model_2.pt", map_location=torch.device('cpu'))
model3 = torch.load(f"{path}/model_3.pt", map_location=torch.device('cpu'))
model4 = torch.load(f"{path}/model_4.pt", map_location=torch.device('cpu'))

sd0 = model0["model_state_dict"]
sd1 = model1["model_state_dict"]
sd2 = model2["model_state_dict"]
sd3 = model3["model_state_dict"]
sd4 = model4["model_state_dict"]
print(sd0.keys())
sd = [sd1, sd2, sd3, sd4]
print(sd0.keys())
wm0 = sd0["skip_connections.0.type_output.weights"]
wm1 = sd0["skip_connections.1.type_output.weights"]
wm2 = sd0["skip_connections.2.type_output.weights"]
wm3 = sd0["skip_connections.3.type_output.weights"]
wm4 = sd0["skip_connections.4.type_output.weights"]
for s in sd:
    wm0 += s["skip_connections.0.type_output.weights"]
    wm1 += s["skip_connections.1.type_output.weights"]
    wm2 += s["skip_connections.2.type_output.weights"]
    wm3 += s["skip_connections.3.type_output.weights"]
    wm4 += s["skip_connections.4.type_output.weights"]
print(torch.softmax(wm0, dim=0))
print(torch.softmax(wm1, dim=0))
print(torch.softmax(wm2, dim=0))
print(torch.softmax(wm3, dim=0))
print(torch.softmax(wm4, dim=0))

