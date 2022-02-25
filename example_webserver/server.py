import asyncio
import threading
from enum import Enum

import uvicorn
import nest_asyncio
from starlette.websockets import WebSocket
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tkinter as tk
from tkinter import ttk

PORT = 8000
HOST = "127.0.0.1"


class MainApplication(tk.Tk):
    """Main GUI for the application"""

    def btn_callback(self):
        """callback function for the reset_ui button."""
        self.reset_ui = True
        self.label['text'] = "No Selection"

    def __init__(self):
        super().__init__()

        self.reset_ui = False

        self.geometry('300x200')
        self.resizable(False, False)
        self.title('Body part selection')

        self.label = ttk.Label(self, text="No selection")
        self.label.pack()

        self.button = ttk.Button(
            self,
            text='Reset Selection',
            command=self.btn_callback
        )
        self.button.pack(
            ipadx=5,
            ipady=5,
            expand=True
        )


class SelectionEnum(str, Enum):
    """structure to group the selections"""
    left_hand = "left_hand"
    right_hand = "right_hand"
    left_foot = "left_foot"
    right_foot = "right_foot"


if __name__ == "__main__":
    # GUI setup
    main_application = MainApplication()

    # server setup
    app = FastAPI(title="Body part selection")
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates/", )


    @app.get("/")
    async def entry_point(request: Request):
        """get method to deliver the index.html file
        :param request: the request from the client"""

        return templates.TemplateResponse(
            "index.html", {"request": request}
        )


    @app.post("/select_body_part", )
    def select_body_part(
            selection: SelectionEnum,
    ) -> None:
        """post method. It receives the selection from the client and changes the label in the GUI
        :param selection: the selection from the client
        """
        if selection == SelectionEnum.left_hand:
            main_application.label['text'] = "Left hand"
        elif selection == SelectionEnum.right_hand:
            main_application.label['text'] = "Right hand"
        elif selection == SelectionEnum.left_foot:
            main_application.label['text'] = "Left foot"
        elif selection == SelectionEnum.right_foot:
            main_application.label['text'] = "Right foot"
        else:
            print("wrong argument")

        return


    @app.websocket("/websocket_reset_selection")
    async def websocket_endpoint(websocket: WebSocket):
        """
        websocket method. If the reset_ui flag is set, the websocket will send a message to the client to reset the button selection.
        :param websocket:
        :return:
        """
        await websocket.accept()

        while True:
            await asyncio.sleep(0.1)
            if main_application.reset_ui:
                await websocket.send_json({"message": "reset_ui"})
                main_application.reset_ui = False

    # Allows the server to be run in this interactive environment
    nest_asyncio.apply()

    class Worker(threading.Thread):
        def run(self):
            # Spin up the server!
            uvicorn.run(app, host=HOST, port=PORT)

    # start the server in a thread
    w = Worker()
    w.start()

    # start the GUI
    main_application.mainloop()
