#!/usr/bin/python3
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from functools import partial
from threading import Thread
from tornado import gen
import pickle


class Epochsviz:
    """This is a simple high level visualization Class built 
    on top of Bokeh in order to visualize training 
    and validation losses during the training (in real time).
    This Class was tested using PyTorch."""

    def __init__(self, title='figure', plot_width=600, plot_height=600):

        self.thread = None
        self.source = ColumnDataSource(data={'epochs': [],
                                             'trainlosses': [],
                                             'vallosses': []}
                                       )
        self.title = title
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.name_train_curve = 'Training loss'
        self.name_val_curve = 'Validation loss'
        self.color_train = 'red'
        self.color_val = 'green'
        self.line_width_train = 2
        self.line_width_val = 2

        self.plot = figure(plot_width=self.plot_width,
                           plot_height=self.plot_height,
                           title=self.title)

        self.plot.line(x='epochs', y='trainlosses',
                       color=self.color_train, legend=self.name_train_curve,
                       line_width=self.line_width_train, alpha=0.8,
                       source=self.source)

        self.plot.line(x='epochs', y='vallosses',
                       color=self.color_val, legend=self.name_val_curve,
                       line_width=self.line_width_val, alpha=0.8,
                       source=self.source)
        self.doc = curdoc()
        self.doc.add_root(self.plot)

    @gen.coroutine
    def update(self, new_data: dict) -> None:
        self.source.stream(new_data)

    def send_data(self,
                  current_epoch: int,
                  current_train_loss: float, current_val_loss: float) -> None:

        new_data = {'epochs': [current_epoch],
                    'trainlosses': [current_train_loss],
                    'vallosses': [current_val_loss]}

        self.doc.add_next_tick_callback(partial(self.update, new_data))

    def start_thread(self, train_function):
        self.thread = Thread(target=train_function)
        self.thread.start()
