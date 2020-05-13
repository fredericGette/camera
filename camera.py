# $env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\frede\dev\camera\camera-key.json"
import io
import os
import numpy as np
import cv2

import asyncio
import json
import websockets

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Message for the websochet clients
MESSAGE = "Hello"

# Connection from a websocket client
async def incomingWSConnection(websocket, path):
    print(f'Connection {websocket} {path}')
    try:
        while True:
            print(f'Send message {MESSAGE} to {websocket}')
            await websocket.send(MESSAGE)
            
            # Wait 10 seconds
            await asyncio.sleep(10)
            
    finally:
        print(f'Deconnection {websocket}')

# Instantiates a vision's client
client = vision.ImageAnnotatorClient()

# Open video capture
cap = cv2.VideoCapture(0)

async def periodic():
    while True:
        print('periodic')

        # Capture frame-by-frame
        ret, frame = cap.read()
        print(f'capture frame {ret}')

        # Convert frame to jpeg
        img_str = cv2.imencode('.jpg', frame)[1].tostring()
        image = types.Image(content=img_str)

        # Call vision API
        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        # Get texts from the API response
        texts = response.text_annotations

        MESSAGE = ''
        print('Texts:')
        for text in texts:
            MESSAGE += '\n"{}"'.format(text.description))

            
            print('\n"{}"'.format(text.description))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                            for vertex in text.bounding_poly.vertices])

            print('bounds: {}'.format(','.join(vertices)))

            # Draw lines
            cv2.line(frame,
                     (text.bounding_poly.vertices[0].x,text.bounding_poly.vertices[0].y),
                     (text.bounding_poly.vertices[1].x,text.bounding_poly.vertices[1].y),
                     (0,255,0), 1)

            cv2.line(frame,
                     (text.bounding_poly.vertices[1].x,text.bounding_poly.vertices[1].y),
                     (text.bounding_poly.vertices[2].x,text.bounding_poly.vertices[2].y),
                     (0,255,0), 1)

            cv2.line(frame,
                     (text.bounding_poly.vertices[2].x,text.bounding_poly.vertices[2].y),
                     (text.bounding_poly.vertices[3].x,text.bounding_poly.vertices[3].y),
                     (0,255,0), 1)

            cv2.line(frame,
                     (text.bounding_poly.vertices[3].x,text.bounding_poly.vertices[3].y),
                     (text.bounding_poly.vertices[0].x,text.bounding_poly.vertices[0].y),
                     (0,255,0), 1)


        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.waitKey(1) # required to display frame

        # Wait 10 seconds
        await asyncio.sleep(10)


print('get event loop')
loop = asyncio.get_event_loop()

try:
    print('start websocket server')
    start_server = websockets.serve(incomingWSConnection, "localhost", 8080)
    loop.run_until_complete(start_server)

    print('start periodic loop')
    task = loop.create_task(periodic())
    loop.run_until_complete(task)

    print('run_forever')
    asyncio.get_event_loop().run_forever()
finally:
    print('after run_forever')
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


