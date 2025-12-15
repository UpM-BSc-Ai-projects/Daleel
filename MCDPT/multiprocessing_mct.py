import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Process,Lock,Value
from .deep_sort_plus.deepsortplus import DeepSortPlus
from boxmot import BoostTrack, BotSort, StrongSort
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct,FieldCondition, MatchValue, Filter
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)
from ultralytics import YOLO
import os
from boxmot.appearance.reid import auto_backend
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
import torch.functional as F

device = "cuda:0"

# import  pygame
from torchvision import transforms
from time import sleep
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index=0)
# device='cuda'
print(device)
print(torch.cuda.get_device_name(0)) 


# sct_config = {
#     'match_threshold': 0.4,
#     'n_clusters': 10,
#     'clust_init_dis_thresh': 0.2,
#     'time_window': 7,
#     'iou_dist_thresh' : 0.8,
#     'merge_thresh':0.4,
#     'rectify_thresh':0.3,
#     'budget': 20
# }
sct_config = {
    'num_clusters': 10,
    'clust_init_dis_thresh': 0.2,
    'time_window': 7,
    'merge_thresh':0.6,
    'rectify_thresh':0.3,
    'stable_time_thresh':5

}







class OSNet(nn.Module):
    def __init__(self, model):
        
        super().__init__()
        self.model=model.cuda()
        self.data_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((256, 128)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                              ])
    def forward(self,input):
       

        batch = torch.cat([self.data_transform(image).unsqueeze(0).cuda() for image in input],dim=0)
        osnet_emb = self.model(batch.cuda()).cpu().numpy()
        norms = np.linalg.norm(osnet_emb,axis=1,keepdims=True)
        return osnet_emb / norms


detector = YOLO(r"rtdetr-l.pt") 
detector.to(device).eval()

def process_stream(vid_path: str, cam_id: int, lock, db_lock=None, Pid=None):
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(r"C:\Users\hthek\OneDrive\Desktop\Collected Dataset\Vid3_cam"+str(cam_id)+".mp4",fourcc=fourcc,fps=25.0,frameSize=(1440,710))

    
    
    client = QdrantClient("http://localhost:6333")
    reid_weights = Path(r"C:\Users\themi\PycharmProjects\Capstone-AI-SE\MCDPT\osnet_x0_25_msmt17.pt")
    osnet = auto_backend.ReidAutoBackend(
                weights=reid_weights, device=device, half=False
            ).model.model.to(device)
    reid_model = OSNet(osnet)
  
    video_path = vid_path
    cap = cv2.VideoCapture(video_path)
    count=0
    tracker = DeepSortPlus(reid_model=reid_model,cam_id=cam_id,global_match_thresh=0.4,sct_config=sct_config)

    with torch.inference_mode():
        while True:
            success, frame = cap.read()
            if not success:
                break

            


            # Run detection
            if count%2==0:
                with lock:
                    torch.cuda.synchronize()
                    output = detector.predict(frame,verbose=False)[0].boxes
                    torch.cuda.synchronize()

                labels = output.cls.cpu().numpy()
                only_people = labels == 0
                scores = output.conf.cpu().numpy()
                scores = scores[only_people]
                keep = scores >= 0.5
                
            
                boxes = output.xyxy.cpu().numpy()[only_people][keep].astype(np.int32)
                
            
            with lock:
                torch.cuda.synchronize()
                tracker.process(frame,[boxes])
                torch.cuda.synchronize()
            for obj in tracker.get_tracked_objects_feat():
                if obj.display and obj.feats:
                    x1,y1,x2,y2 = map(int,obj.rect)
                    global_id = obj.track_id
                    
                    max_ppl = 15
                    norm = mpl.colors.Normalize(vmin=0, vmax=max_ppl)
                    cmap = cm.hot
                    m = cm.ScalarMappable(norm=norm, cmap=cmap)
                    color_num = max_ppl - (int(global_id.split("-")[0]) % max_ppl)
                    color = m.to_rgba(color_num)[:3]*np.array([255]).astype(np.int32)

                    
                    

                    cv2.rectangle(frame,(x1,y1),(x2,y2),color=color,thickness=2)
                    cv2.putText(frame,"id: "+global_id,(x1,y1-5),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=color)
                    with db_lock:

                        points = []
                        for i,feat in enumerate(obj.feats):
                            points.append(
                                    PointStruct(
                                        id = Pid.value + i,
                                        vector = feat,
                                        payload = {"Pid": global_id ,"coords" : [x1,y1,x2,y2],
                                                   "cam_id":cam_id, "frame_count":count,"cross_cam":obj.track.cross_camera_track},
                                                )
                                        )
                            
                        client.upsert(
                        collection_name="stream",
                        points = points
                                    )
                        
            # out.write(frame)
                        Pid.value+=len(points)

            
            count+=1
            
        

            
          
            cv2.imshow('BoXMOT + Torchvision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
          
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    lock = Lock()
    db_lock = Lock()
    Pid = Value('i',0)


    t1 = Process(target=process_stream
                 ,kwargs={"vid_path": r"C:\Users\hthek\OneDrive\Desktop\Collected Dataset\Vid_3_cam_0.mp4"
                         ,"cam_id":0
                        ,"lock":lock
                        ,"db_lock":db_lock
                        ,"Pid":Pid})
    t2 = Process(target=process_stream
                 ,kwargs={"vid_path": r"C:\Users\hthek\OneDrive\Desktop\Collected Dataset\Vid_3_cam_1.mp4"
                        ,"cam_id": 1
                        ,"lock":lock
                        ,"db_lock":db_lock
                        ,"Pid":Pid}
                        )

    t1.start()
    t2.start()

    t1.join()
    t2.join()
        