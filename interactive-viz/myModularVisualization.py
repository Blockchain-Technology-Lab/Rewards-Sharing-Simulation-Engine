from mesa.visualization.ModularVisualization import PageHandler
from mesa.visualization.ModularVisualization import SocketHandler
from mesa.visualization.UserParam import UserSettableParameter

import os
import tornado.escape
import webbrowser

class MyPageHandler(PageHandler):
    def get(self):
        elements = self.application.visualization_elements
        for i, element in enumerate(elements):
            element.index = i
        self.render(
            "my_modular_template.html",
            port=self.application.port,
            model_name=self.application.model_name,
            description=self.application.description,
            package_includes=self.application.package_includes,
            local_includes=self.application.local_includes,
            scripts=self.application.js_code,
        )

class MySocketHandler(SocketHandler):
    """
    Same as mesa's SocketHandler, with the exception that it
     automatically resets the model when the user changes the parameters
    """

    def on_message(self, message):
        """Receiving a message from the websocket, parse, and act accordingly."""
        if self.application.verbose:
            print(message)
        msg = tornado.escape.json_decode(message)

        if msg["type"] == "get_step":
            if not self.application.model.running:
                self.write_message({"type": "end"})
            else:
                self.application.model.step()
                self.write_message(self.viz_state_message)

        elif msg["type"] == "reset":
            self.application.reset_model()
            self.write_message(self.viz_state_message)

        elif msg["type"] == "submit_params":
            param = msg["param"]
            value = msg["value"]

            # Is the param editable?
            if param in self.application.user_params:
                if isinstance(
                    self.application.model_kwargs[param], UserSettableParameter
                ):
                    self.application.model_kwargs[param].value = value
                else:
                    self.application.model_kwargs[param] = value
                self.application.reset_model()

        else:
            if self.application.verbose:
                print("Unexpected message!")

#taken from  mesa.visualization.ModularVisualization
class MyModularServer(tornado.web.Application):
    """Main visualization application."""

    verbose = True

    port = int(os.getenv("PORT", 8521))  # Default port to listen on
    max_steps = 100000

    # Handlers and other globals:
    page_handler = (r"/", MyPageHandler)
    socket_handler = (r"/ws", MySocketHandler)
    static_handler = (
        r"/static/(.*)",
        tornado.web.StaticFileHandler,
        {"path": os.path.dirname(__file__) + "/templates"},
    )
    local_handler = (r"/local/(.*)", tornado.web.StaticFileHandler, {"path": ""})

    handlers = [page_handler, socket_handler, static_handler, local_handler]

    settings = {
        "debug": True,
        "autoreload": False,
        "template_path": os.path.dirname(__file__) + "/templates",
    }

    EXCLUDE_LIST = ("width", "height")

    def __init__(
        self, model_cls, visualization_elements, name="Mesa Model", model_params={}
    ):
        """Create a new visualization server with the given elements."""
        # Prep visualization elements:
        self.visualization_elements = visualization_elements
        self.package_includes = set()
        self.local_includes = set()
        self.js_code = []
        for element in self.visualization_elements:
            for include_file in element.package_includes:
                self.package_includes.add(include_file)
            for include_file in element.local_includes:
                self.local_includes.add(include_file)
            self.js_code.append(element.js_code)

        # Initializing the model
        self.model_name = name
        self.model_cls = model_cls
        self.description = "No description available"
        if hasattr(model_cls, "description"):
            self.description = model_cls.description
        elif model_cls.__doc__ is not None:
            self.description = model_cls.__doc__

        self.model_kwargs = model_params
        self.reset_model()

        # Initializing the application itself:
        super().__init__(self.handlers, **self.settings)

    @property
    def user_params(self):
        result = {}
        for param, val in self.model_kwargs.items():
            if isinstance(val, UserSettableParameter):
                result[param] = val.json

        return result

    def reset_model(self):
        """Reinstantiate the model object, using the current parameters."""

        model_params = {}
        for key, val in self.model_kwargs.items():
            if isinstance(val, UserSettableParameter):
                if (
                    val.param_type == "static_text"
                ):  # static_text is never used for setting params
                    continue
                model_params[key] = val.value
            else:
                model_params[key] = val

        self.model = self.model_cls(**model_params)

    def render_model(self):
        """Turn the current state of the model into a dictionary of
        visualizations

        """
        visualization_state = []
        for element in self.visualization_elements:
            element_state = element.render(self.model)
            visualization_state.append(element_state)
        return visualization_state

    def launch(self, port=None, open_browser=True):
        """Run the app."""
        if port is not None:
            self.port = port
        url = "http://127.0.0.1:{PORT}".format(PORT=self.port)
        print("Interface starting at {url}".format(url=url))
        self.listen(self.port)
        if open_browser:
            webbrowser.open(url)
        tornado.autoreload.start()
        tornado.ioloop.IOLoop.current().start()