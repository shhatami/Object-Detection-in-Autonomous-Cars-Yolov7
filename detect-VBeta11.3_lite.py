import os, numpy as np, cv2, psutil, GPUtil, math, csv, time, re, random, subprocess, platform, logging, threading, multiprocessing

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
from datetime import datetime
from torch import nn
import torch, torch.backends.cudnn as cudnn, torch.multiprocessing as mp, torchvision



logger = logging.getLogger(__name__)


def start(save__log):
    Time = datetime.now()
    print(" starting data logger...")
    print (Time.strftime("%Y-%m-%d_%H-%M-%S"))
    print(platform.system())
    print(platform.release())
    print(platform._sys_version())
    if save__log == True:
        with open("Log.txt","a") as f:
            writer = csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([Time.strftime("%Y-%m-%d_%H-%M-%S_")+ str(':Starting') , ])
	
    print(" starting video stream...")

def storage(j,save__log):
    Time = datetime.now()
    if platform.system() == 'Linux':
        while True:
            try:
                path = 'detected'
                path = os.path.expanduser("detected/cam-" + str(j) + "/." + Time.strftime("%Y-%m-%d"))
                if not os.path.exists(path):
                    os.makedirs(path) 
            except: 
                print("Storage-System-Malfunction")
                #cv2.putText(im0,"Storage-System-Malfunction", (850, 120), cv2.FONT_HERSHEY_SIMPLEX,1, (50,105,128), 3,cv2.LINE_AA)
                if save__log == True:
                    with open("Log.txt","a") as f:
                        writer = csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([Time.strftime("%Y-%m-%d_%H-%M-%S_")+ str("Storage-System-Malfunction") , ])
                continue
    
            return path 

    elif platform.system() == 'Windows': 
        while True:
            try:
                path = 'detected'
                path = os.path.expanduser("detected\cam-" + str(j) + "\." + Time.strftime("%Y-%m-%d"))
                if not os.path.exists(path):
                    os.makedirs(path) 
            except: 
                print("Storage-System-Malfunction")
                #cv2.putText(im0,"Storage-System-Malfunction", (850, 120), cv2.FONT_HERSHEY_SIMPLEX,1, (50,105,128), 3,cv2.LINE_AA)
                if save__log == True:
                    with open("Log.txt","a") as f:
                        writer = csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([Time.strftime("%Y-%m-%d_%H-%M-%S_")+ str("Storage-System-Malfunction") , ])
                continue
    
            return path 

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
        return math.ceil(x / divisor) * divisor
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def git_describe(path=Path(__file__).parent): 

    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return '' 

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')
    
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def centroid(xyxy,conf,cls,gn):
    
    #xywh = (xywh2xyxy(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
    #line = (cls, *xywh, conf)     
    #print(('%g ' * len(line)).rstrip() % line + '\n')            
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    center1 = (x1 + x2) // 2
    center2 = (y1 + y2) // 2
    center = (center1, center2)

    return center, center1, center2, x2, y2

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def box_iou(box1, box2):


    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):


    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def draws(draw,xyxy,conf,cls,gn):
    if draw == True :
        center, center1, center2 ,x2, y2  = centroid(xyxy,conf,cls,gn)

def hud(overlays,time_overlay,performance,save__log,im0,det,s):
    if overlays == True:
        cv2.putText(im0, '{}'.format(s), (40, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(im0, 'Total detection : {}'.format(len(det)), (40, 110), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3, cv2.LINE_AA)
        if time_overlay == True:
            cv2.putText(im0,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), (800, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3, cv2.LINE_AA)
        if performance == True:
            cv2.putText(im0, 'CPU :{}'.format(psutil.cpu_percent(1)), (1600, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA) 
            cv2.putText(im0, 'RAM %:{}'.format(psutil.virtual_memory().percent), (1600, 110), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA) 
            #cv2.putText(im0, 'RAM :{}'.format(psutil.virtual_memory()[3]/1000000000), (1600, 210), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA)
            cv2.putText(im0, 'GPU % :{}'.format(GPUtil.showUtilization()), (1600, 160), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA)
            if save__log == True:
                with open("Log.txt","a") as f:
                    writer = csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+ str(psutil.cpu_percent(1)) , ])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+ str(psutil.virtual_memory().percent) , ])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+ str(psutil.virtual_memory()[3]/1000000000) , ])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+ str(GPUtil.showUtilization()) , ])

def check_imshow():
    # Check if environment supports image displays
    def isdocker():
    # Is environment a Docker container
        return Path('/workspace').exists()  # or Path('/.dockerenv').exists()
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False

def get_video(source):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))
    return webcam

def fps(show_fps, im0, s, t1, t2, t3):
    if show_fps == True:            
        #fps = (1E3 * (t2 - t1) / 1000)
        #fps0 = (1E3 * (t3 - t2) / 1000)
        #print('FPS : ',fps)  
        print(f' ({(1E3 * (t2 - t1)):.1f}ms) Inference') 
        print(f' ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        cv2.putText(im0, 'FPS NMS :{}'.format(1E3 * (t3 - t2)), (1600, 210), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA)
        cv2.putText(im0, 'FPS Inference :{}'.format(1E3 * (t2 - t1)), (1600, 210), cv2.FONT_HERSHEY_SIMPLEX,1, (224,225,224), 3,cv2.LINE_AA)
 
def vid_save(save_img, dataset, im0, vid_cap, project):
    if save_img == True:
        if dataset.mode == 'image':
            cv2.imwrite(project, im0)
            print(f" The image with the result is saved in: {project}")
        else:  # 'video' or 'stream'
            if vid_path != project:  # new video
                vid_path = project
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap == True:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    project += '.mp4'
                vid_writer = cv2.VideoWriter(project, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

def view(view__img, im0, p):
    if view__img == True:
        cv2.imshow(str(p), cv2.resize(im0, (800, 600)))
        cv2.waitKey(1)

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|#¡·$€%&()=?¿^*,¨´><+]", repl="_", string=s)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def autopad(k, p=None):  
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

class LoadStreams:  
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
                
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = threading.Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def detect(source, device, view__img, save_img, save__log, vid_cap, save_detect, save_dir):

    #Detection Parameters
    imgsz = 640
    iou_thres = 0.45
    conf_thres = 0.25
    classes = (0,2,3,5,7,9)
    colors = (133,45,160)
    if platform.system() == 'Linux':
        weights = 'weights/yolov7-tiny.pt'
    if platform.system() == 'Windows':
        weights = 'weights\yolov7-tiny.pt'
    project= save_dir  
    augment = ''
    agnostic = ''
    half = False
    draw = True
    overlays = True
    performance = False
    time_overlay = True
    show_fps = True

    start(save__log)
    #set_logging()

    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location = device) 
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride)  

    if half == True:
        model.half() 
    
    vid_path, vid_writer = None, None
    if get_video(source):
        
        view_img = check_imshow()
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        #dataset = LoadWebcam(source, img_size=imgsz, stride=stride)
    #else:
        #dataset = LoadImages(source, img_size=imgsz, stride=stride)

    
    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)
        t3 = time_synchronized()
       
        for i, det in enumerate(pred):  

            if get_video(source):  
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 

            if len(det):               
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                founded_classes={}
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    class_index=int(c) 
                    founded_classes[names[class_index]]=int(n) 
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                                
                for *xyxy, conf, cls in reversed(det): 
                    
                    draws(draw,xyxy,conf,cls,gn)

                    if view_img == True:  
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors, line_thickness=1)

                        
                     
            with open('streams1.txt', 'r') as f:
                file = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
                NoE = len(file)
           

            

            #fps(show_fps, im0, s, t1, t2, t3)

            hud(overlays,time_overlay,performance,save__log,im0,det,s)

            view(view__img, im0, p)

            vid_save(save_img, dataset, im0, vid_cap, project)
            
 
if __name__ == '__main__':     

    processes = []
    p0 = multiprocessing.Process(target=detect,
    args=('streams1.txt','cpu',         #video Adress & device
        True,False,False,False,False,    #view_img, save_img, save_log, vid_cap, save_detect, relays, relay_sn
        storage(1,True)))                   

    
    p0.start()

