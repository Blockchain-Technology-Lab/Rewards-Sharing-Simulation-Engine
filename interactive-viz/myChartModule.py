from mesa.visualization.modules.ChartVisualization import ChartModule


class MyChartModule(ChartModule):

    def render(self, model):
        current_values = []
        data_collector = getattr(model, self.data_collector_name)

        for s in self.series:
            name = s["Label"]
            try:
                val = data_collector.model_vars[name][-1]  # Latest value
            except (IndexError, KeyError):
                continue #todo maybe add sth to know if it happens any time other than the beginning
            current_values.append(val)
        return current_values
