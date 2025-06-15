#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
vision_detection_node.py
    Subscribes : /camera/image_undistorted   (sensor_msgs/CompressedImage)
    Publishes  : /obj_detect/person_detect   (final.msg/DetectedPerson)

"""

import os, cv2, numpy as np, rospy, time
from sensor_msgs.msg import CompressedImage, RegionOfInterest
from final.msg import DetectedPerson
import tensorrt as trt, pycuda.driver as cuda, numpy as np
import pycuda.autoinit


class TrtDetector(object):
    def __init__(self, engine_path):

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        for idx in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(idx)
            shape = self.ctx.get_binding_shape(idx)
            rospy.loginfo("TensorRT binding %d: %-20s %s", idx, name, shape)



        self.stream      = cuda.Stream()
        self.bindings    = [None] * self.engine.num_bindings 
        self.device_mem  = [None] * self.engine.num_bindings 
        self.host_out    = []                                     
        self.out_idx     = []                                     

        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                in_shape = (1, 3, 300, 300)                 
                self.ctx.set_binding_shape(idx, in_shape)
                size = trt.volume(in_shape) * np.float32().nbytes
            else:
                shape = tuple(self.ctx.get_binding_shape(idx))
                size  = trt.volume(shape) * np.float32().nbytes
                self.out_idx.append(idx)
                self.host_out.append(np.empty(shape, dtype=np.float32))

            self.device_mem[idx] = cuda.mem_alloc(size)
            self.bindings[idx]   = int(self.device_mem[idx])

        self.input_idx = [i for i in range(self.engine.num_bindings)
                          if self.engine.binding_is_input(i)][0]
        self.d_input   = self.device_mem[self.input_idx]

    def _preproc(self, frame):
        img = cv2.resize(frame, (300, 300))
        img = (img.astype(np.float32) - 127.5) * 0.007843
        blob = img.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob) 

    def infer(self, frame):

        blob = self._preproc(frame)
        cuda.memcpy_htod_async(self.d_input, blob, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)

        # copy every output back to host
        for host, idx in zip(self.host_out, self.out_idx):
            cuda.memcpy_dtoh_async(host, self.device_mem[idx], self.stream)
        self.stream.synchronize()

        for host in self.host_out:
            if host.ndim == 3 and host.shape[-1] == 7:
                return host.reshape(-1, 7)

        boxes = scores = classes = num = None
        for arr in self.host_out:
            shp = arr.shape
            if shp[-1] == 4 and arr.ndim == 3:              
                boxes = arr
            elif shp[-1] == 1 and arr.size == 1:            
                num = arr
            elif arr.ndim == 2 and shp[0] == 1:             
                if np.all(arr <= 1.0):
                    scores = arr
                else:
                    classes = arr

        if None in (boxes, scores, classes):
            raise RuntimeError("TrtDetector: could not recognise engine outputs")

        n = int(num[0]) if num is not None else scores.shape[1]
        # build [imgID, cls, conf, x1, y1, x2, y2]
        det = np.column_stack([
            np.zeros(n, dtype=np.float32),          # fake imgID = 0
            classes[0, :n].astype(np.float32),
            scores [0, :n],
            boxes  [0, :n],
        ])
        
        return det

class VisionDetectionNode(object):
    def __init__(self):
        rospy.init_node('vision_detection_node', anonymous=True)
        rospy.loginfo("VisionDetection: CUDA/TensorRT backend initialising…")

        script_dir  = os.path.dirname(os.path.realpath(__file__))
        engine_path = os.path.join(script_dir, "ssd_mobilenet_v2.trt")
        if not os.path.isfile(engine_path):
            rospy.logfatal("Missing %s - run trtexec first!", engine_path)
            raise SystemExit(1)
        rospy.loginfo("ik ben bij 1")
        self.detector = TrtDetector(engine_path)

        self.pers_pub = rospy.Publisher(
            '/obj_detect/person_detect', DetectedPerson, queue_size=1)
        rospy.loginfo("ik ben bij 2")
        rospy.Subscriber('/camera/image_undistorted',
                         CompressedImage,
                         self.image_cb,
                         queue_size=1,
                         buff_size=2**24)

        self.person_cls  = 15
        self.conf_thresh = 0.20
        rospy.loginfo("ik ben bij 3")

    def image_cb(self, msg):
        rospy.loginfo("ik ben bij 4")
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                                 cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr("VisionDetection: JPEG decode failed: %s", e)
            return
        rospy.loginfo("ik ben bij 5")
        t0 = time.time()
        detections = self.detector.infer(frame)
        infer_ms = (time.time() - t0) * 1000.0
        rospy.loginfo("Inference %.1f ms, detections = %d",
                      infer_ms, detections.shape[0])
        rospy.loginfo("ik ben bij 6")
        h, w = frame.shape[:2]
        for det in detections:
            cls, conf = int(det[1]), det[2]
            if cls != self.person_cls or conf < self.conf_thresh:
                continue
            rospy.loginfo("ik ben bij 7")
            x1 = int(det[3] * w); y1 = int(det[4] * h)
            x2 = int(det[5] * w); y2 = int(det[6] * h)

            x1, x2 = sorted((max(0, x1), max(0, x2)))
            y1, y2 = sorted((max(0, y1), max(0, y2)))
            if x2 <= x1 or y2 <= y1:
                rospy.logwarn("Skipping invalid bbox [%d,%d]-[%d,%d]", x1, y1, x2, y2)
                continue

            crop = frame[y1:y2, x1:x2].copy()
            ok, jpeg = cv2.imencode('.jpg', crop)
            if not ok:
                rospy.logwarn("JPEG re-encode failed for bbox [%d,%d]-[%d,%d]", x1, y1, x2, y2)
                continue

            crop_msg            = CompressedImage()
            crop_msg.header.stamp = rospy.Time.now()
            crop_msg.format     = "jpeg"
            crop_msg.data       = jpeg.tobytes()

            bb_msg              = RegionOfInterest()
            bb_msg.x_offset, bb_msg.y_offset = x1, y1
            bb_msg.width,  bb_msg.height     = x2 - x1, y2 - y1

            person_msg          = DetectedPerson()
            person_msg.image    = crop_msg
            person_msg.bbox     = bb_msg

            self.pers_pub.publish(person_msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        VisionDetectionNode().spin()
    except rospy.ROSInterruptException:
        pass
