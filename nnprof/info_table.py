#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import tabulate

from torch.autograd.profiler import format_time
try:
    from torch.autograd.profiler import format_memory
except ImportError:
    from .utils import format_memory


class InfoTable:

    def __init__(self, headers, data, with_percent=False):
        """
        Args:
            header (Iterable[string]): header of info table.
            data (Iterable[numpy.array]): data of table.
            with_percent (bool): whether data presented with percent data or not.
        """
        assert len(headers) == len(data), "length of headers and data are not matched"
        self.headers = headers
        self.info = {key: value for key, value in zip(headers, data)}
        self.with_percent = with_percent

    def insert(self, header, data, position=-1):
        """
        insert header and data into current table at given position

        Args:
            header (Iterable[string]): inserted data.
            data (Iterable[numpy.array]): inserted data.
            position (int): insert position, the same usage like list indexing.
        """

        def swap(a, b):
            a, b = b, a

        self.info[header] = data
        if header in self.headers:
            index = self.headers.index(header)
            swap(self.headers[index], self.headers[position])
        else:
            self.headers.insert(position, header)

    def sorted_by(self, keyname=None, descending=True):
        """
        use keyname to sort table.

        Args:
            keyname (string): sorted header name.
            descending (bool): whether sorted in descending order or not.
        """
        if keyname is None:
            return self
        if keyname not in self.info:
            keyname += "_avg"
        assert keyname in self.info
        sort_index = np.argsort(self.info[keyname], axis=0).reshape(-1)
        if descending:
            sort_index = sort_index[::-1]
        for header in self.headers:
            self.info[header] = self.info[header][sort_index]

        return self

    def filter(self, filter_list=None):
        """
        filter header and data in filter list.

        Args:
            filter_list (Iterable[string]): list of headers that needs to be filtered out
        """
        self.headers = [header for header in self.headers if header not in filter_list]

    def filter_zeros(self):
        """filter all zeros data."""
        filter_list = []
        for header in self.headers:
            data = self.info[header]
            if "float" in data.dtype.name or "int" in data.dtype.name:
                if data.sum() == 0:
                    filter_list.append(header)
        self.filter(filter_list)

    def average(self, average_key="hits"):
        """
        average table with average key

        Args:
            average_key:
        """
        hits = self.info[average_key]
        for i, header in enumerate(self.headers):
            if header.endswith("time") or header.endswith("mem"):
                self.info[header + "_avg"] = self.info[header] / hits
                self.headers[i] += "_avg"
                del self.info[header]

    def row_limit(self, limit=None):
        """
        return table in row limit format.

        Args:
            limit (int): row limit number of table, None means no limit.
        """
        if limit is None:
            return self
        else:
            data = [self.info[x][:limit] for x in self.headers]
            return InfoTable(headers=self.headers, data=data)

    def __str__(self):
        self.filter_zeros()
        time_formatter = np.vectorize(format_time)
        mem_formatter = np.vectorize(format_memory)
        percent_formatter = np.vectorize(lambda x: " ({:.2%})".format(x))

        fmt_data = []
        for header in self.headers:
            data = self.info[header]
            if "time" in header:
                time_array = time_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data.sum())
                    time_array = np.core.defchararray.add(time_array, percent)
                fmt_data.append(time_array)
            elif "mem" in header:
                mem_array = mem_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data.sum())
                    mem_array = np.core.defchararray.add(mem_array, percent)
                fmt_data.append(mem_array)
            else:
                fmt_data.append(data)

        concat_data = np.concatenate(fmt_data, axis=1)
        table = tabulate.tabulate(concat_data, headers=self.headers, tablefmt="fancy_grid")
        return table


class TreeTable(InfoTable):

    def __init__(self, headers, data, with_percent=False, max_depth=3):
        super().__init__(headers, data, with_percent)
        self.max_depth = max_depth

    def __str__(self):
        self.filter_zeros()
        time_formatter = np.vectorize(format_time)
        mem_formatter = np.vectorize(format_memory)
        percent_formatter = np.vectorize(lambda x: " ({:.2%})".format(x))

        fmt_data = []
        for header in self.headers:
            data = self.info[header]
            if "time" in header:
                time_array = time_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data[0])  # data[0] is the sum value
                    time_array = np.core.defchararray.add(time_array, percent)
                fmt_data.append(time_array)
            elif "mem" in header:
                mem_array = mem_formatter(data)
                if self.with_percent:
                    percent = percent_formatter(data / data[0])  # sum value, ditto
                    mem_array = np.core.defchararray.add(mem_array, percent)
                fmt_data.append(mem_array)
            else:
                fmt_data.append(data)

        concat_data = np.concatenate(fmt_data, axis=1)

        # white space should be kept under profile
        old_ws = tabulate.PRESERVE_WHITESPACE
        tabulate.PRESERVE_WHITESPACE = True
        table = tabulate.tabulate(concat_data, headers=self.headers, tablefmt="fancy_grid")
        tabulate.PRESERVE_WHITESPACE = old_ws
        return table
