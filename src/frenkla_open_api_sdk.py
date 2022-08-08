import datetime as dt
import requests


class FrenklaOpenApiService(object):
    def __init__(self, base_url: str = "https://api.frenkla.com/public/api/"):
        """Instantiates an object of the class FrenklaOpenApiService

        Parameters
        ----------
        base_url: str
            Web address of the Frenkla API
        """
        self.headers = {"accept": "application/json"}
        self.base_url = base_url

    def __repr__(self):

        return "{}(base_url={})".format(self.__class__.__name__, self.base_url)

    def get_timeseries(self, timeseries_id: str, url: str = "/opentimeseries") -> dict:
        """Endpoint to fetch a timeseries

        Parameters
        ----------
        timeseries_id: str
            The Frenkla database id of the required timeseries
        url: str, default: /opentimeseries
            Web address of the Frenkla API endpoint

        Returns
        -------
        dict
            A timeseries
        """

        response = requests.get(
            url=self.base_url + f"{url}/{timeseries_id}", headers=self.headers
        )

        if response.status_code // 100 != 2:
            raise Exception(f"{response.status_code}, {response.text}")

        return response.json()

    def get_fundinfo(
        self, isins: list, report_date: dt.date | None = None, url: str = "/fundinfo"
    ) -> dict:
        """Endpoint to fetch information about Captor funds

        Parameters
        ----------
        isins: list
            The Frenkla database id of the required timeseries
        report_date: datetime.date, optional
            Date variable
        url: str, default: /fundinfo
            Web address of the Frenkla API endpoint

        Returns
        -------
        dict
            Fund information
        """

        params = {"isins": isins}
        if report_date:
            params.update({"reportDate": report_date.strftime("%Y-%m-%d")})
        response = requests.get(
            url=self.base_url + url, params=params, headers=self.headers
        )

        if response.status_code // 100 != 2:
            raise Exception(f"{response.status_code}, {response.text}")

        return response.json()

    def get_nav(self, isin: str, url: str = "/nav") -> dict:
        """Endpoint to fetch NAV data of a Captor fund

        Parameters
        ----------
        isin: str
            ISIN code of the required Captor fund
        url: str, default: /nav
            Web address of the Frenkla API endpoint

        Returns
        -------
        dict
            Fund information
        """

        response = requests.get(url=self.base_url + url, headers=self.headers)

        if response.status_code // 100 != 2:
            raise Exception(f"{response.status_code}, {response.text}")

        output = {}
        result = response.json()
        for res in result:
            if res["isin"] == isin:
                output.update(res)

        if len(output) == 0:
            raise Exception(
                f"Request for NAV series using ISIN {isin} returned no data."
            )

        return output
