"""
Chart Module
============

Module for drawing live-updating scatter charts using Charts.js

"""
import json
from mesa.visualization.ModularVisualization import VisualizationElement

class BubbleChartModule(VisualizationElement):
    """
    Attributes:
        series: A list of dictionaries containing information on series to
                plot. Each dictionary must contain (at least) the "Label" and
                "Color" keys. The "Label" value must correspond to a
                model-level series collected by the model's DataCollector, and
                "Color" must have a valid HTML color.
        canvas_height, canvas_width: The width and height to draw the chart on
                                     the page, in pixels. Default to 200 x 500
        data_collector_name: Name of the DataCollector object in the model to
                             retrieve data from.
        template: "chart_module.html" stores the HTML template for the module.

    """

    package_includes = ["Chart.min.js"]
    local_includes = ["BubbleChartModule.js"]

    def __init__(
        self,
        series,
        canvas_height=200,
        canvas_width=500,
        data_collector_name="datacollector",
    ):
        """
        Create a new scatter chart visualization.

        Args:
            series: A list of dictionaries containing series names and
                    HTML colors to chart them in, e.g.
                    [{"Label": "happy", "Color": "Black"},]
            canvas_height, canvas_width: Size in pixels of the chart to draw.
            data_collector_name: Name of the DataCollector to use.
        """

        self.series = series
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.data_collector_name = data_collector_name

        series_json = json.dumps(self.series)
        new_element = "new BubbleChartModule({}, {},  {})"
        new_element = new_element.format(series_json, canvas_width, canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        current_values = []
        data_collector = getattr(model, self.data_collector_name)

        for s in self.series:
            name = s["Label"]
            try:
                data_dict = data_collector.model_vars[name][-1]  # Latest values collected
            except (IndexError, KeyError):
                continue
            x = data_dict['x']
            y = data_dict['y']
            r = data_dict['r']
            pool_ids = data_dict['pool_id']
            owner_ids = data_dict['owner_id']
            for i in range(len(x)):
                current_values.append(x[i])
                current_values.append(y[i])
                current_values.append(r[i])
                current_values.append(pool_ids[i])
                current_values.append(owner_ids[i])
        return current_values
