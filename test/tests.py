import datetime as dt
import json
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.tseries.offsets import CDay
import sys
import unittest

from OpenSeries.series import OpenTimeSeries, timeseries_chain
from OpenSeries.frame import OpenFrame, key_value_table
from OpenSeries.sim_price import ReturnSimulation
from OpenSeries.risk import cvar_down, var_down
from OpenSeries.datefixer import date_fix

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if repo_root not in sys.path:
    sys.path.append(repo_root)


def sim_to_opentimeseries(sim: ReturnSimulation, end: dt.date) -> OpenTimeSeries:
    date_range = pd.date_range(periods=sim.trading_days, end=end, freq=CDay(calendar=OpenTimeSeries.sweden))
    sdf = sim.df.iloc[0].T.to_frame()
    sdf.index = date_range
    sdf.columns = pd.MultiIndex.from_product([['Asset'], ['Return(Total)']])
    return OpenTimeSeries.from_df(sdf, valuetype='Return(Total)')


def sim_to_openframe(sim: ReturnSimulation, end: dt.date) -> OpenFrame:
    date_range = pd.date_range(periods=sim.trading_days, end=end, freq=CDay(calendar=OpenTimeSeries.sweden))
    tslist = []
    for item in range(sim.number_of_sims):
        sdf = sim.df.iloc[item].T.to_frame()
        sdf.index = date_range
        sdf.columns = pd.MultiIndex.from_product([[f'Asset_{item}'], ['Return(Total)']])
        tslist.append(OpenTimeSeries.from_df(sdf, valuetype='Return(Total)'))
    return OpenFrame(tslist)


class TestOpenTimeSeries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        OpenTimeSeries.setup_class()

        cls.sim = ReturnSimulation.from_merton_jump_gbm(n=1, d=2512,
                                                        mu=0.05,
                                                        vol=0.1,
                                                        jumps_lamda=0.00125,
                                                        jumps_sigma=0.001,
                                                        jumps_mu=-0.2,
                                                        seed=71)

        cls.randomseries = sim_to_opentimeseries(cls.sim, end=dt.date(2019, 6, 30)).to_cumret()
        cls.random_properties = cls.randomseries.all_properties().to_dict()[('Asset', 'Price(Close)')]

    def test_opentimeseries_tsdf_not_empty(self):

        json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'series.json')
        with open(json_file, 'r') as ff:
            output = json.load(ff)
        timeseries = OpenTimeSeries(output)

        self.assertFalse(timeseries.tsdf.empty)

    def test_create_opentimeseries_from_open_api(self):

        timeseries_id = '59977d91f3fa6319ecb41cbd'
        timeseries = OpenTimeSeries.from_open_api(timeseries_id=timeseries_id)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

    def test_create_opentimeseries_from_open_nav(self):

        fund = 'SE0009807308'
        timeseries = OpenTimeSeries.from_open_nav(isin=fund)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

    def test_create_opentimeseries_from_open_fundinfo(self):

        fund = 'SE0009807308'
        timeseries = OpenTimeSeries.from_open_fundinfo(isin=fund)

        self.assertTrue(isinstance(timeseries, OpenTimeSeries))

    def test_create_opentimeseries_from_pandas_df(self):

        se = pd.Series(data=[1.0, 1.01, 0.99, 1.015, 1.003],
                       index=['2019-06-24', '2019-06-25', '2019-06-26', '2019-06-27', '2019-06-28'],
                       name='Asset_0')
        df = pd.DataFrame(data=[[1.0, 1.0], [1.01, 0.98], [0.99, 1.004], [1.015, 0.976], [1.003, 0.982]],
                          index=['2019-06-24', '2019-06-25', '2019-06-26', '2019-06-27', '2019-06-28'],
                          columns=['Asset_0', 'Asset_1'])

        seseries = OpenTimeSeries.from_df(df=se)
        dfseries = OpenTimeSeries.from_df(df=df, column_nmbr=1)

        self.assertTrue(isinstance(seseries, OpenTimeSeries))
        self.assertTrue(isinstance(dfseries, OpenTimeSeries))

    def test_create_opentimeseries_from_frame(self):

        sim_f = ReturnSimulation.from_merton_jump_gbm(n=2, d=2512, mu=0.05, vol=0.1,
                                                      jumps_lamda=0.00125, jumps_sigma=0.001, jumps_mu=-0.2, seed=71)
        frame_f = sim_to_openframe(sim_f, end=dt.date(2019, 6, 30))
        frame_f.to_cumret()
        fseries = OpenTimeSeries.from_frame(frame_f, label='Asset_1')

        self.assertTrue(isinstance(fseries, OpenTimeSeries))

    def test_save_opentimeseries_to_json(self):

        seriesfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'irisc.json')
        capirisc = 'SE0009807308'
        irisc = OpenTimeSeries.from_open_nav(isin=capirisc)
        irisc.to_json(filename=seriesfile)

        self.assertTrue(os.path.exists(seriesfile))

        os.remove(seriesfile)

        self.assertFalse(os.path.exists(seriesfile))

    def test_create_opentimeseries_from_fixed_rate(self):

        fixseries = OpenTimeSeries.from_fixed_rate(rate=0.03, days=756, end_dt=dt.date(2019, 6, 30))

        self.assertTrue(isinstance(fixseries, OpenTimeSeries))

    def test_warn_on_jsonschema_when_create_opentimeseries_from_dict(self):

        fund = 'SE0009807308'
        timeseries1 = OpenTimeSeries.from_open_nav(isin=fund)

        new_dict = timeseries1.attributes
        cleaner_list = ['local_ccy', 'tsdf']  # 'local_ccy' not removed to trigger ValidationError
        for item in cleaner_list:
            new_dict.pop(item)

        with self.assertRaises(Exception):
            OpenTimeSeries(new_dict)

        new_dict.pop('label')
        new_dict['dates'] = []  # Set dates attribute to empty array to trigger minItems ValidationError

        with self.assertRaises(Exception):
            OpenTimeSeries(new_dict)

    def test_opentimeseries_periods_in_a_year(self):

        calc = len(self.randomseries.dates) / \
               ((date_fix(self.randomseries.dates[-1]) - date_fix(self.randomseries.dates[0])).days / 365.25)

        self.assertEqual(calc, self.randomseries.periods_in_a_year)
        self.assertEqual(f'{251.3720547945205:.13f}', f'{self.randomseries.periods_in_a_year:.13f}')
        all_prop = self.random_properties['periods_in_a_year']
        self.assertEqual(f'{all_prop:.13f}', f'{self.randomseries.periods_in_a_year:.13f}')

    def test_opentimeseries_yearfrac(self):

        self.assertEqual(f'{9.9931553730322:.13f}', f'{self.randomseries.yearfrac:.13f}')
        all_prop = self.random_properties['yearfrac']
        self.assertEqual(f'{all_prop:.13f}', f'{self.randomseries.yearfrac:.13f}')

    def test_opentimeseries_resample(self):

        rs_sim = ReturnSimulation.from_merton_jump_gbm(n=1, d=2512, mu=0.05, vol=0.1,
                                                       jumps_lamda=0.00125, jumps_sigma=0.001, jumps_mu=-0.2, seed=71)
        rs_series = sim_to_opentimeseries(rs_sim, end=dt.date(2019, 6, 30)).to_cumret()

        before = rs_series.value_ret

        rs_series.resample(freq='BM')

        self.assertEqual(121, rs_series.length)
        self.assertEqual(before, rs_series.value_ret)

    def test_opentimeseries_nan_nandf(self):

        simnan = ReturnSimulation.from_merton_jump_gbm(n=1, d=2512, mu=0.05, vol=0.1,
                                                       jumps_lamda=0.00125, jumps_sigma=0.001, jumps_mu=-0.2, seed=71)
        nanseries = sim_to_opentimeseries(simnan, end=dt.date(2019, 6, 30)).to_cumret()

        self.assertFalse(nanseries.nan)

        nanseries.tsdf.iloc[1] = None

        self.assertTrue(nanseries.nan)
        self.assertEqual('2009-07-01', nanseries.nandf.index[0].strftime('%Y-%m-%d'))

        nanseries.value_nan_handle()
        self.assertFalse(nanseries.nan)

        nanseries.value_to_ret()
        nanseries.tsdf.iloc[1] = None

        self.assertTrue(nanseries.nan)
        self.assertEqual('2009-07-01', nanseries.nandf.index[0].strftime('%Y-%m-%d'))

        nanseries.return_nan_handle()
        self.assertFalse(nanseries.nan)

    def test_opentimeseries_calc_range(self):

        csim = ReturnSimulation.from_normal(n=1, d=1200, mu=0.05, vol=0.1, seed=71)
        cseries = sim_to_opentimeseries(csim, end=dt.date(2019, 6, 30)).to_cumret()

        dates = cseries.calc_range(months_offset=48)

        self.assertListEqual(['2015-06-26', '2019-06-28'],
                             [dates[0].strftime('%Y-%m-%d'), dates[1].strftime('%Y-%m-%d')])
        dates = self.randomseries.calc_range(from_dt=dt.date(2016, 6, 30))

        self.assertListEqual(['2016-06-30', '2019-06-28'],
                             [dates[0].strftime('%Y-%m-%d'), dates[1].strftime('%Y-%m-%d')])

        gr_0 = cseries.vol_func(months_from_last=48)

        cseries.dates = cseries.dates[-1008:]
        cseries.values = cseries.values[-1008:]
        cseries.pandas_df()
        cseries.set_new_label(lvl_one='Return(Total)')
        cseries.to_cumret()

        gr_1 = cseries.vol

        self.assertEqual(f'{gr_0:.13f}', f'{gr_1:.13f}')

    def test_opentimeseries_value_to_diff(self):

        diffsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        diffseries = sim_to_opentimeseries(diffsim, end=dt.date(2019, 6, 30)).to_cumret()
        diffseries.value_to_diff()
        are_bes = [f'{nn[0]:.12f}' for nn in diffseries.tsdf.values]
        should_bes = ['0.000000000000', '-0.007322627296', '-0.002581366067', '0.003248920666', '-0.002628519782',
                      '0.003851856296', '0.007573468698', '-0.005893167569', '0.001567531620', '-0.005246297149',
                      '-0.001822686581', '0.009014775004', '-0.004289844249', '-0.008344628763', '-0.010412377959']

        self.assertListEqual(are_bes, should_bes)

    def test_opentimeseries_value_to_ret(self):

        retsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        retseries = sim_to_opentimeseries(retsim, end=dt.date(2019, 6, 30)).to_cumret()

        retseries.value_to_ret(logret=False)
        are_bes = [f'{nn[0]:.12f}' for nn in retseries.tsdf.values]
        should_bes = ['0.000000000000', '-0.007322627296', '-0.002600407884', '0.003281419826', '-0.002646129969',
                      '0.003887950443', '0.007614830448', '-0.005880572955', '0.001573434257', '-0.005257779632',
                      '-0.001836330888', '0.009098966716', '-0.004290865956', '-0.008382584742', '-0.010548160048']

        self.assertListEqual(are_bes, should_bes)

        retseries.to_cumret()

        retseries.value_to_ret(logret=True)
        are_log = [f'{nn[0]:.12f}' for nn in retseries.tsdf.values]
        should_log = ['0.000000000000', '-0.007349569336', '-0.002603794818', '0.003276047717', '-0.002649637160',
                      '0.003880411897', '0.007585983975', '-0.005897931610', '0.001572197706', '-0.005271650396',
                      '-0.001838019010', '0.009057820522', '-0.004300098140', '-0.008417916189', '-0.010604186218']

        self.assertListEqual(are_log, should_log)

    def test_opentimeseries_log_and_exp(self):

        logsim = ReturnSimulation.from_normal(n=1, d=15, mu=0.05, vol=0.1, seed=71)
        logseries = sim_to_opentimeseries(logsim, end=dt.date(2019, 6, 30)).to_cumret()
        b4_log = [f'{nn[0]:.12f}' for nn in logseries.tsdf.values]

        logseries.value_to_log()
        are_log = [f'{nn[0]:.12f}' for nn in logseries.tsdf.values]

        logseries.value_to_log(reverse=True)
        are_exp = [f'{nn[0]:.12f}' for nn in logseries.tsdf.values]

        should_log = ['0.000000000000', '-0.007349569336', '-0.009953364154', '-0.006677316437', '-0.009326953597',
                      '-0.005446541699', '0.002139442275', '-0.003758489335', '-0.002186291629', '-0.007457942025',
                      '-0.009295961036', '-0.000238140514', '-0.004538238654', '-0.012956154844', '-0.023560341062']

        should_exp = ['1.000000000000', '0.992677372704', '0.990096006637', '0.993344927303', '0.990716407521',
                      '0.994568263817', '1.002141732515', '0.996248564946', '0.997816096566', '0.992569799417',
                      '0.990747112836', '0.999761887839', '0.995472043590', '0.987127414827', '0.976715036868']

        self.assertListEqual(are_log, should_log)
        self.assertListEqual(are_exp, should_exp)
        self.assertListEqual(b4_log, are_exp)

    def test_all_calc_properties(self):

        checks = {'arithmetic_ret': f'{0.00242035119:.11f}',
                  'cvar_down': f'{-0.01402077271:.11f}',
                  'geo_ret': f'{0.00242231676:.11f}',
                  'kurtosis': f'{180.63357183510:.11f}',
                  'max_drawdown': f'{-0.40011625413:.11f}',
                  'max_drawdown_cal_year': f'{-0.23811167802:.11f}',
                  'positive_share': f'{0.49940262843:.11f}',
                  'ret_vol_ratio': f'{0.02071179512:.11f}',
                  'skew': f'{-6.94679906059:.11f}',
                  'twr_ret': f'{0.00241939932:.11f}',
                  'value_ret': f'{0.02447195802:.11f}',
                  'var_down': f'{-0.01059129607:.11f}',
                  'vol': f'{0.11695349153:.11f}',
                  'vol_from_var': f'{0.10208932904:.11f}',
                  'worst': f'{-0.19174232326:.11f}',
                  'worst_month': f'{-0.19165644070:.11f}',
                  'z_score': f'{1.21195350537:.11f}'}
        for c in checks:
            self.assertEqual(checks[c], f'{getattr(self.randomseries, c):.11f}', msg=f'Difference in: {c}')
            self.assertEqual(f'{self.random_properties[c]:.11f}', f'{getattr(self.randomseries, c):.11f}',
                             msg=f'Difference in: {c}')

    def test_all_calc_functions(self):

        checks = {'arithmetic_ret_func': f'{0.003481792158:.12f}',
                  'cvar_down_func': f'{-0.013318898358:.12f}',
                  'geo_ret_func': f'{0.003484394437:.12f}',
                  'kurtosis_func': f'{-0.161645660276:.12f}',
                  'max_drawdown_func': f'{-0.205657752819:.12f}',
                  'positive_share_func': f'{0.506454816286:.12f}',
                  'ret_vol_ratio_func': f'{0.033606022349:.12f}',
                  'skew_func': f'{-0.036159475308:.12f}',
                  'twr_ret_func': f'{0.003478362038:.12f}',
                  'value_ret_func': f'{0.014029906514:.12f}',
                  'var_down_func': f'{-0.010958301720:.12f}',
                  'vol_func': f'{0.103683631486:.12f}',
                  'vol_from_var_func': f'{0.105686426193:.12f}',
                  'worst_func': f'{-0.020634872447:.12f}',
                  'z_score_func': f'{1.368253357728:.12f}'}
        for c in checks:
            self.assertEqual(checks[c], f'{getattr(self.randomseries, c)(months_from_last=48):.12f}',
                             msg=f'Difference in {c}')

        self.assertEqual(f'{0.076502833914:.12f}',
                         f'{getattr(self.randomseries, "value_ret_calendar_period")(year=2019):.12f}')

    def test_opentimeseries_max_drawdown_date(self):

        self.assertEqual(dt.date(2018, 11, 8), self.randomseries.max_drawdown_date)
        all_prop = self.random_properties['max_drawdown_date']
        self.assertEqual(all_prop, self.randomseries.max_drawdown_date)

    def test_openframe_max_drawdown_date(self):

        mddsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        mddframe = sim_to_openframe(mddsim, dt.date(2019, 6, 30)).to_cumret()
        self.assertListEqual([dt.date(2018, 8, 15), dt.date(2018, 7, 2), dt.date(2018, 8, 3), dt.date(2018, 10, 3),
                              dt.date(2018, 10, 17)], mddframe.max_drawdown_date.tolist())

    def test_opentimeseries_running_adjustment(self):

        simadj = ReturnSimulation.from_merton_jump_gbm(n=1, d=2512, mu=0.05, vol=0.1,
                                                       jumps_lamda=0.00125, jumps_sigma=0.001, jumps_mu=-0.2, seed=71)
        adjustedseries = sim_to_opentimeseries(simadj, end=dt.date(2019, 6, 30)).to_cumret()
        adjustedseries.running_adjustment(0.05)

        self.assertEqual(f'{1.689055852583:.12f}', f'{float(adjustedseries.tsdf.iloc[-1]):.12f}')

    @staticmethod
    def create_list_randomseries(num_series: int) -> list:

        sims = []
        np.random.seed(71)
        for g in range(num_series):
            sim_0 = ReturnSimulation.from_normal(n=1, d=100, mu=0.05, vol=0.1, seed=None)
            series = sim_to_opentimeseries(sim_0, end=dt.date(2019, 6, 30))
            series.set_new_label(lvl_zero=f'Asset_{g}')
            sims.append(series)
        return sims

    def test_returnsimulation_toframe_vs_toseries(self):

        n = 10
        frame_0 = OpenFrame(self.create_list_randomseries(n)).to_cumret()
        dict_toseries = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=n, d=100, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))
        frame_1.to_cumret()
        dict_toframe = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toseries, dict_toframe)

    def test_openframe_add_timeseries(self):

        n = 4
        sims = self.create_list_randomseries(n)

        frame_0 = OpenFrame(sims[:-1])
        frame_0.add_timeseries(sims[-1])

        dict_toseries = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=4, d=100, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))

        dict_toframe = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toseries, dict_toframe)

    def test_openframe_delete_timeseries(self):

        dsim = ReturnSimulation.from_normal(n=4, d=100, mu=0.05, vol=0.1, seed=71)

        frame = sim_to_openframe(dsim, end=dt.date(2019, 6, 30))
        frame.weights = [0.4, 0.1, 0.2, 0.3]

        lbl = 'Asset_1'
        frame.delete_timeseries(lbl)
        labels = [ff.label for ff in frame.constituents]

        self.assertListEqual(labels, ['Asset_0', 'Asset_2', 'Asset_3'])
        self.assertListEqual(frame.weights, [0.4, 0.2, 0.3])

    def test_openframe_tocumret_and_back_toret(self):

        fmt = '{:.12f}'

        sim_0 = ReturnSimulation.from_normal(n=4, d=61, mu=0.05, vol=0.1, seed=71)
        frame_0 = sim_to_openframe(sim_0, end=dt.date(2019, 6, 30))

        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.to_cumret()
        frame_0.value_to_ret()
        frame_0.tsdf = frame_0.tsdf.applymap(lambda x: fmt.format(x))

        dict_toframe_0 = frame_0.tsdf.to_dict()

        sim_1 = ReturnSimulation.from_normal(n=4, d=61, mu=0.05, vol=0.1, seed=71)
        frame_1 = sim_to_openframe(sim_1, end=dt.date(2019, 6, 30))

        # The below adjustment is not ideal but I believe I implemented it to mimic behaviour of Bbg return series.
        frame_1.tsdf.iloc[0] = 0

        frame_1.tsdf = frame_1.tsdf.applymap(lambda x: fmt.format(x))

        dict_toframe_1 = frame_1.tsdf.to_dict()

        self.assertDictEqual(dict_toframe_0, dict_toframe_1)

    def test_openframe_nan_nandf(self):

        sim_nanframe = ReturnSimulation.from_normal(n=4, d=61, mu=0.05, vol=0.1, seed=71)
        frame_nan = sim_to_openframe(sim_nanframe, end=dt.date(2019, 6, 30))

        self.assertFalse(frame_nan.nan)

        frame_nan.tsdf.iloc[1, 1] = None

        self.assertTrue(frame_nan.nan)
        self.assertEqual('2019-03-29', frame_nan.nandf.index[0].strftime('%Y-%m-%d'))

        frame_nan.tsdf.iloc[1, 1] = 0.01
        self.assertFalse(frame_nan.nan)

        frame_nan.tsdf.iloc[1, 1] = np.nan

        self.assertTrue(frame_nan.nan)
        self.assertEqual('2019-03-29', frame_nan.nandf.index[0].strftime('%Y-%m-%d'))

    def test_openframe_keyvaluetable_with_relative_results(self):

        json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'key_value_table_with_relative.json')
        with open(json_file, 'r', encoding='utf-8') as ff:
            output = json.load(ff)

        sim_rel = ReturnSimulation.from_normal(n=4, d=2512, mu=0.05, vol=0.1, seed=71)
        frame_rel = sim_to_openframe(sim_rel, end=dt.date(2019, 6, 30)).to_cumret()
        frame_rel.relative(base_zero=False)

        kv = key_value_table(frame_rel)
        fmt = '{:.11f}'
        kv = kv.applymap(lambda x: fmt.format(x))
        dd = kv.to_dict(orient='index')
        new_dd = {str(k): dd[k] for k in dd}

        self.assertDictEqual(new_dd, output)

    def test_risk_functions_same_as_series_and_frame_methods(self):

        riskdata = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)

        riskseries = sim_to_opentimeseries(riskdata, end=dt.date(2019, 6, 30))
        riskseries.set_new_label(lvl_zero='Asset_0')
        riskframe = sim_to_openframe(riskdata, end=dt.date(2019, 6, 30))
        riskseries.to_cumret()
        riskframe.to_cumret()

        self.assertEqual(riskseries.cvar_down, cvar_down(riskseries.tsdf.iloc[:, 0].tolist()),
                         msg='CVaR for OpenTimeSeries not equal')
        self.assertEqual(riskseries.var_down, var_down(riskseries.tsdf.iloc[:, 0].tolist()),
                         msg='VaR for OpenTimeSeries not equal')

        self.assertEqual(riskframe.cvar_down.iloc[0], cvar_down(riskframe.tsdf.iloc[:, 0]),
                         msg='CVaR for OpenFrame not equal')
        self.assertEqual(riskframe.var_down.iloc[0], var_down(riskframe.tsdf.iloc[:, 0]),
                         msg='VaR for OpenFrame not equal')

    def test_openframe_methods_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)

        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30))
        sameseries.set_new_label(lvl_zero='Asset_0')
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30))
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        methods = ['rolling_return', 'rolling_vol', 'rolling_var_down', 'rolling_cvar_down']
        for method in methods:
            assert_frame_equal(getattr(sameseries, method)(), getattr(sameframe, method)(column=0))

        cumseries = sameseries.from_deepcopy()
        cumframe = sameframe.from_deepcopy()

        sameseries.value_to_log()
        sameframe.value_to_log()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.value_to_log(reverse=True)
        sameframe.value_to_log(reverse=True)
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        assert_frame_equal(sameseries.tsdf, cumseries.tsdf)
        assert_frame_equal(cumframe.tsdf, sameframe.tsdf)

        sameseries.value_to_ret()
        sameframe.value_to_ret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.to_cumret()
        sameframe.to_cumret()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.resample()
        sameframe.resample()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

        sameseries.value_to_diff()
        sameframe.value_to_diff()
        assert_frame_equal(sameseries.tsdf, sameframe.tsdf)

    def test_openframe_calc_methods_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=2, d=504, mu=0.05, vol=0.175, seed=71)

        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero='Asset_0')
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        methods_to_compare = ['arithmetic_ret_func', 'cvar_down_func', 'geo_ret_func', 'kurtosis_func',
                              'max_drawdown_func', 'positive_share_func', 'ret_vol_ratio_func', 'skew_func',
                              'target_weight_from_var', 'twr_ret_func', 'value_ret_func', 'var_down_func',
                              'vol_from_var_func', 'vol_func', 'worst_func', 'z_score_func']
        for m in methods_to_compare:
            self.assertEqual(f'{getattr(sameseries, m)(months_from_last=12):.11f}',
                             f'{float(getattr(sameframe, m)(months_from_last=12).iloc[0]):.11f}')

    def test_opentimeseries_measures_same_as_openframe_measures(self):

        sims = []
        np.random.seed(71)
        for g in range(10):
            sim_0 = ReturnSimulation.from_normal(n=1, d=100, mu=0.05, vol=0.1, seed=None)
            series = sim_to_opentimeseries(sim_0, end=dt.date(2019, 6, 30))
            series.set_new_label(lvl_zero=f'Asset_{g}')
            series.to_cumret()
            sims.append(series)
        frame_0 = OpenFrame(sims).to_cumret()
        common_calc_props = ['arithmetic_ret', 'cvar_down', 'geo_ret', 'kurtosis', 'max_drawdown',
                             'max_drawdown_cal_year', 'positive_share', 'ret_vol_ratio', 'skew', 'twr_ret', 'value_ret',
                             'var_down', 'vol', 'vol_from_var', 'worst', 'worst_month', 'z_score']
        series_measures = []
        frame_measures = []
        for p in common_calc_props:
            fr = getattr(frame_0, p).tolist()
            fr = [f'{ff:.10f}' for ff in fr]
            frame_measures.append(fr)
            se = [f'{getattr(s, p):.10f}' for s in sims]
            series_measures.append(se)

        self.assertListEqual(series_measures, frame_measures)

    def test_openframe_properties_same_as_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=504, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero='Asset_0')
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

        common_props_to_compare = ['periods_in_a_year', 'yearfrac']
        for c in common_props_to_compare:
            self.assertEqual(getattr(sameseries, c), getattr(sameframe, c))

    def test_keeping_attributes_aligned_openframe_vs_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=255, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero='Asset_0')
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

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

        series_props = [a for a in dir(sameseries) if not a.startswith('__') and not callable(getattr(sameseries, a))]
        series_compared = set(series_props).symmetric_difference(set(common_calc_props + common_props +
                                                                     common_attributes + series_attributes))
        self.assertTrue(len(series_compared) == 0, msg=f'Difference is: {series_compared}')
        frame_props = [a for a in dir(sameframe) if not a.startswith('__') and not callable(getattr(sameframe, a))]
        frame_compared = set(frame_props).symmetric_difference(set(common_calc_props + common_props +
                                                                   common_attributes + frame_attributes +
                                                                   frame_calc_props))
        self.assertTrue(len(frame_compared) == 0, msg=f'Difference is: {frame_compared}')

    def test_keeping_methods_aligned_openframe_vs_opentimeseries(self):

        same = ReturnSimulation.from_normal(n=1, d=255, mu=0.05, vol=0.175, seed=71)
        sameseries = sim_to_opentimeseries(same, end=dt.date(2019, 6, 30)).to_cumret()
        sameseries.set_new_label(lvl_zero='Asset_0')
        sameframe = sim_to_openframe(same, end=dt.date(2019, 6, 30)).to_cumret()

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

        series_methods = [a for a in dir(sameseries) if not a.startswith('__') and callable(getattr(sameseries, a))]
        series_compared = set(series_methods).symmetric_difference(set(common_calc_methods + common_methods +
                                                                       series_createmethods + series_unique))
        self.assertTrue(len(series_compared) == 0, msg=f'Difference is: {series_compared}')

        frame_methods = [a for a in dir(sameframe) if not a.startswith('__') and callable(getattr(sameframe, a))]
        frame_compared = set(frame_methods).symmetric_difference(
            set(common_calc_methods + common_methods + frame_unique))
        self.assertTrue(len(frame_compared) == 0, msg=f'Difference is: {frame_compared}')

    def test_openframe_log_and_exp(self):

        logsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        logframe = sim_to_openframe(logsim, dt.date(2019, 6, 30)).to_cumret()

        aa = logframe.tsdf.applymap(lambda nn: f'{nn:.12f}')
        bb = aa.to_dict(orient='list')
        b4_log = [bb[k] for k in bb]

        logframe.value_to_log()

        aa = logframe.tsdf.applymap(lambda nn: f'{nn:.12f}')
        bb = aa.to_dict(orient='list')
        middle_log = [bb[k] for k in bb]

        self.assertNotEqual(b4_log, middle_log)

        logframe.value_to_log(reverse=True)

        aa = logframe.tsdf.applymap(lambda nn: f'{nn:.12f}')
        bb = aa.to_dict(orient='list')
        after_log = [bb[k] for k in bb]

        self.assertListEqual(b4_log, after_log)

    def test_openframe_correl_matrix(self):

        corrsim = ReturnSimulation.from_normal(n=5, d=252, mu=0.05, vol=0.1, seed=71)
        corrframe = sim_to_openframe(corrsim, dt.date(2019, 6, 30)).to_cumret()
        dict1 = corrframe.correl_matrix.applymap(lambda nn: f'{nn:.12f}').to_dict()
        dict2 = {'Asset_0': {'Asset_0': '1.000000000000', 'Asset_1': '0.008448597235', 'Asset_2': '0.059458117640',
                             'Asset_3': '0.071395739932', 'Asset_4': '0.087545728279'},
                 'Asset_1': {'Asset_0': '0.008448597235', 'Asset_1': '1.000000000000', 'Asset_2': '-0.040605114787',
                             'Asset_3': '0.030023445985', 'Asset_4': '0.074249393671'},
                 'Asset_2': {'Asset_0': '0.059458117640', 'Asset_1': '-0.040605114787', 'Asset_2': '1.000000000000',
                             'Asset_3': '-0.015715823407', 'Asset_4': '0.064477746560'},
                 'Asset_3': {'Asset_0': '0.071395739932', 'Asset_1': '0.030023445985', 'Asset_2': '-0.015715823407',
                             'Asset_3': '1.000000000000', 'Asset_4': '0.038405133612'},
                 'Asset_4': {'Asset_0': '0.087545728279', 'Asset_1': '0.074249393671', 'Asset_2': '0.064477746560',
                             'Asset_3': '0.038405133612', 'Asset_4': '1.000000000000'}}

        self.assertDictEqual(dict1, dict2)

    def test_timeseries_chain(self):

        full_sim = ReturnSimulation.from_normal(n=1, d=252, mu=0.05, vol=0.1, seed=71)
        full_series = sim_to_opentimeseries(full_sim, end=dt.date(2019, 6, 30)).to_cumret()
        full_values = [f'{nn:.10f}' for nn in full_series.tsdf.iloc[:, 0].tolist()]

        front_series = OpenTimeSeries.from_df(full_series.tsdf.iloc[:126])

        back_series = OpenTimeSeries.from_df(full_series.tsdf.loc[front_series.last_idx:])

        chained_series = timeseries_chain(front_series, back_series)
        chained_values = [f'{nn:.10f}' for nn in chained_series.values]

        self.assertListEqual(full_series.dates, chained_series.dates)
        self.assertListEqual(full_values, chained_values)
