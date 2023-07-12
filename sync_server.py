import time
from threading import Thread
import uvicorn
from vidgear.gears.asyncio import WebGear
import cv2
from vidgear.gears.asyncio.helper import reducer
from vidgear.gears import NetGear
import collections
import asyncio
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.responses import StreamingResponse


#global_frame = collections.deque(maxlen=100)
client_options = {"bidirectional_mode": True}
#global_message = collections.deque(maxlen=100)
# 创建一个同时支持异步和线程安全的队列
return_message = "None"
client = NetGear(receive_mode=True, logging=False, port="13000", **client_options)

'''
def get_frames_proc(frame_queue, message_queue):
    while True:
        data = client.recv()

        # Check if data is None
        if data is None:
            continue

        # Unpack the received data
        extra_message, frame = data

        # Check if frame is None
        if frame is None:
            continue

        frame_queue.append(frame)
        message_queue.append(extra_message)  # 将extra_message保存到队列中


async def message_producer():
    global return_message

    while True:
        try:
            yield return_message  # 生成消息
            await asyncio.sleep(0)
        except IndexError:
            await asyncio.sleep(0)
            # time.sleep(0.00001)
'''

async def my_frame_producer():
    global return_message
        
    while True:
        try:
            # 使用异步方法从队列中获取帧
            #frame = await frame_queue.async_q.get()
            #return_message = await message_queue.async_q.get()

            # frame = await reducer(frame, percentage=0.1)
            #frame = await reducer(
            #    frame, percentage=30, interpolation=cv2.INTER_AREA
            #)  # reduce frame by 30%
            
            data = client.recv()

            # Check if data is None
            if data is None:
                continue

            # Unpack the received data
            extra_message, frame = data
            frame = cv2.resize(frame, (640, 480))

            # Check if frame is None
            if frame is None:
                continue
            encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()

            # yield frame and message in byte format
            yield (b"--frame\r\nContent-Type:image/jpeg2000\r\n\r\n" + encodedImage + b"\r\n")

            await asyncio.sleep(0.00001)
        except (IndexError, Exception) as e:  # Catch IndexError when deque is empty
            await asyncio.sleep(0.00001)
            #time.sleep(0.00001)

'''
async def message_response(scope):
    """
    Return an async message streaming response for `message_producer` generator
    """
    assert scope["type"] in ["http", "https"]
    await asyncio.sleep(0)
    return StreamingResponse(
        message_producer(),
        media_type="text/plain",
    )
'''

if __name__ == '__main__':
    # create janus queues within a running event loop
    '''
    t = Thread(target=get_frames_proc, args=())
    t.start()
    '''
    # close stream

    # add various performance tweaks as usual
    options = {
        # "frame_size_reduction": 0.1,  # 20% reduction
        "frame_jpeg_quality": 90,
        "frame_jpeg_optimize": True,
        "frame_jpeg_progressive": False,
        "custom_data_location": "customdir"
    }

    # initialize WebGear app with a valid source
    web = WebGear(
        logging=False, **options
    )  # enable source i.e. `test.mp4` and enable `logging` for debugging
    # append new route to point our rendered webpage
    web.config["generator"] = my_frame_producer
    #web.routes.append(
    #    Route("/message", endpoint=message_response)
    #)
    # run this app on Uvicorn server at address http://localhost:8000/
    uvicorn.run(web(), host="0.0.0.0", port=8991)

    # close app safely
    web.shutdown()
    # t.join()
