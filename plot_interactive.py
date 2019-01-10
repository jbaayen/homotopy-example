import numpy as np
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.plotting import figure, output_file, show

from example import results, times

theta_values = list(results.keys())
variable_names = results[theta_values[0]].keys()


shared_data = {"times": times, "theta_values": [theta_values for time in times]}
plots = []
sliders = []
for var in variable_names:
    plot_data = shared_data.copy()
    plot_data["y"] = results[1.0][var]
    plot_data["all_results"] = [
        [results[theta][var][tidx] for theta in theta_values]
        for tidx, time in enumerate(times)
    ]
    source = ColumnDataSource(data=plot_data)
    plot = figure(
        title=var,
        plot_width=400,
        plot_height=400,
        y_range=(
            min(np.min(results[theta][var]) for theta in theta_values),
            max(np.max(results[theta][var]) for theta in theta_values),
        ),
    )
    plot.line("times", "y", source=source, line_width=3, line_alpha=0.6)
    plots.append(plot)

    with open("interactive.js", "r") as code_file:
        callback = CustomJS(args=dict(source=source), code=code_file.read())

    slider = Slider(
        start=0.0, end=1.0, value=1, step=0.01, title="theta", callback=callback
    )
    sliders.append(slider)

elements = [column(s, p) for s, p in zip(sliders, plots)]
layout = gridplot(elements, ncols=3)
output_file("interactive.html", title="Homotopy Deformation")

show(layout)
