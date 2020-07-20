# OpenSeries

This is a project where we keep tools to perform timeseries analysis on a single asset or a group of assets.

[OpenTimeSeries](https://github.com/CaptorAB/Python/blob/master/Apps/OpenSeries/series.py) 
is the *Class* for managing and analyzing a single timeseries. 
[OpenFrame](https://github.com/CaptorAB/Python/blob/master/Apps/OpenSeries/frame.py) 
is the *Class* for managing a group of timeseries, and e.g. calculate a portfolio timeseries from 
a rebalancing strategy between timeseries.

## Usage: [OpenTimeSeries](https://github.com/CaptorAB/Python/blob/master/Apps/OpenSeries/series.py)

    # -*- coding: utf-8 -*-
    from Apps.OpenSeries.series import OpenTimeSeries


    if __name__ == '__main__':

        series = OpenTimeSeries.from_open_nav(isin='SE0009807308')
        series.plot_series(tick_fmt='.1%')
        print(series.all_properties())

## Development

First, below is the top-level 
[OpenTimeSeries](https://github.com/CaptorAB/Python/blob/master/Apps/OpenSeries/series.py) 
class attributes defined:  

    class OpenTimeSeries(object):

        _id: str
        instrumentId: str
        currency: str
        dates: List[str]
        domestic: str
        name: str
        isin: str
        label: str
        schema: dict
        sweden: CaptorHolidayCalendar
        valuetype: str
        values: List[float]
        local_ccy: bool
        tsdf: pd.DataFrame

The [OpenTimeSeries](https://github.com/CaptorAB/Python/blob/master/Apps/OpenSeries/series.py) 
\__init\__ method:
    
        schema_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openseries.json')
        with open(file=schema_file, mode='r', encoding='utf-8') as f:
            series_schema = json.load(f)

        try:
            jsonschema.validate(instance=d, schema=series_schema)
        except ValidationError as e:
            raise Exception(d.get('_id', None), d.get('name', None), e)

        self.__dict__ = d

        if self.name != '':
            self.label = self.name

        self.pandas_df()



Each source will have its own classmethod. Calling such method will return a new class object. 
The methods' output will be a dictionary which will then set the object attributes via 
the self.\__dict\__ = d statement.

To easily use all of the tools available in the Pandas library an attribute tsdf is added to 
the class via the method below.
    
    def pandas_df(self):
        """
        This method creates a Pandas DataFrame attribute, tsdf, from the given list of dates and values.
        """
        df = pd.DataFrame(data=self.values, index=self.dates, dtype='float64')
        df.columns = pd.MultiIndex.from_product([[self.label], [self.valuetype]])
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)

        self.tsdf = df

        return self

Listed below are the **attributes** given by the source and calculated **properties**. They are separated into
 *common*, applicable to both OpenTimeSeries & OpenFrame, and those that are specific for the respective classes. 

    common_calc_props = ['arithmetic_ret', 'cvar_down', 'geo_ret', 'kurtosis', 'max_drawdown',
                         'max_drawdown_cal_year', 'positive_share', 'ret_vol_ratio', 'skew', 'twr_ret', 'value_ret',
                         'var_down', 'vol', 'vol_from_var', 'worst', 'worst_month', 'z_score']

    common_props = ['periods_in_a_year', 'yearfrac', 'max_drawdown_date']

    common_attributes = ['attributes', 'length', 'first_idx', 'last_idx', 'nan', 'nandf', 'tsdf', 'sweden']

    series_attributes = ['values', 'local_ccy', '_id', 'instrumentId', 'currency', 'isin', 'dates', 'name',
                         'valuetype', 'label', 'domestic']

    frame_attributes = ['constituents', 'columns_lvl_zero', 'columns_lvl_one', 'item_count', 'labels', 'weights',
                        'first_indices', 'last_indices', 'lengths_of_items']

    frame_calc_props = ['correl_matrix']

Listed below are the **methods** that belong to the OpenTimeSeries & OpenFrame classes.

    common_calc_methods = ['arithmetic_ret_func', 'cvar_down_func', 'geo_ret_func', 'kurtosis_func',
                           'max_drawdown_func', 'positive_share_func', 'ret_vol_ratio_func', 'skew_func',
                           'target_weight_from_var', 'twr_ret_func', 'value_ret_func', 'var_down_func',
                           'vol_from_var_func', 'vol_func', 'worst_func', 'z_score_func']

    common_methods = ['align_index_to_local_cdays', 'all_properties', 'calc_range', 'from_deepcopy', 'plot_series',
                      'resample', 'return_nan_handle', 'rolling_return', 'rolling_vol', 'rolling_cvar_down',
                      'rolling_var_down', 'to_cumret', 'to_drawdown_series', 'value_nan_handle',
                      'value_ret_calendar_period', 'value_to_diff', 'value_to_log', 'value_to_ret']

    series_createmethods = ['from_open_api', 'from_open_nav', 'from_open_fundinfo', 'from_df', 'from_frame',
                            'from_fixed_rate', 'from_quandl']

    series_unique = ['pandas_df', 'running_adjustment', 'set_new_label', 'to_json', 'validate_vs_schema',
                     'setup_class', 'show_public_series_url']

    frame_unique = ['add_timeseries', 'delete_timeseries', 'delete_tsdf_item', 'drawdown_details',
                    'ord_least_squares_fit', 'make_portfolio', 'relative', 'rolling_corr', 'trunc_frame',
                    'warn_diff_nbr_datapoints', 'warn_nan_present']

A TODO is to describe what the above attributes, properties and methods are, but hopefully 
most of the names are self-explanatory.