"""Investigating how to remove multiple inheritance structure."""
from __future__ import annotations

from logging import warning
from typing import Optional

from pydantic import BaseModel, ValidationError


class TopSingle(BaseModel):  # type: ignore[misc, unused-ignore]

    """
    Declare TopSingle.

    Parameters
    ----------
    tsdf: int
        tsdf attribute

    Returns
    -------
    TopSingle
        Object of the class TopSingle
    """

    tsdf: int


class LeftSingle(TopSingle):

    """
    Declare LeftSingle.

    Parameters
    ----------
    tsdf: int
        tsdf attribute
    length: int
        length attribute

    Returns
    -------
    LeftSingle
        Object of the class LeftSingle
    """

    tsdf: int
    length: int


class RightSingle(TopSingle):

    """Declare RightSingle."""

    constituents: tuple[int, int]
    weights: Optional[int] = None
    tsdf: int = 10

    def __init__(
        self: RightSingle,
        constituents: tuple[int, int],
        weights: Optional[int] = None,
    ) -> None:
        """
        Declare RightSingle.

        Parameters
        ----------
        constituents: tuple[int, int]
            constituents attribute
        weights: int, optional
            weights attribute

        Returns
        -------
        RightSingle
            Object of the class RightSingle
        """
        super().__init__(  # type: ignore[call-arg, unused-ignore]
            constituents=constituents,
            weights=weights,
        )
        self.constituents = constituents
        self.weights = weights
        self.set_tsdf()

    def set_tsdf(self: RightSingle) -> None:
        """Set the attribute tsdf."""
        if self.constituents is not None and len(self.constituents) != 0:
            self.tsdf = self.constituents[0]


class TopMulti:

    """
    Declare TopMulti.

    Parameters
    ----------
    tsdf: int
        tsdf attribute

    Returns
    -------
    TopMulti
        Object of the class TopMulti
    """

    tsdf: int


class LeftMulti(BaseModel, TopMulti):  # type: ignore[misc, unused-ignore]

    """
    Declare LeftMulti.

    Parameters
    ----------
    tsdf: int
        tsdf attribute
    length: int
        length attribute

    Returns
    -------
    LeftMulti
        Object of the class LeftMulti
    """

    tsdf: int
    length: int


class RightMulti(BaseModel, TopMulti):  # type: ignore[misc, unused-ignore]

    """Declare RightMulti."""

    constituents: tuple[int, int]
    weights: Optional[int] = None
    tsdf: int = 10

    def __init__(
        self: RightMulti,
        constituents: tuple[int, int],
        weights: Optional[int] = None,
    ) -> None:
        """
        Declare RightMulti.

        Parameters
        ----------
        constituents: tuple[int, int]
            constituents attribute
        weights: int, optional
            weights attribute

        Returns
        -------
        RightMulti
            Object of the class RightMulti
        """
        super().__init__(constituents=constituents, weights=weights)
        self.constituents = constituents
        self.weights = weights
        self.set_tsdf()

    def set_tsdf(self: RightMulti) -> None:
        """Set the attribute tsdf."""
        if self.constituents is not None and len(self.constituents) != 0:
            self.tsdf = self.constituents[0]


if __name__ == "__main__":
    ts1 = TopSingle(tsdf=1)
    print("TopSingle", ts1)  # noqa: T201

    ls1 = LeftSingle(tsdf=1, length=2)
    print("LeftSingle", ls1)  # noqa: T201

    try:
        rs1 = RightSingle(constituents=(1, 2))
        print("RightSingle", rs1)  # noqa: T201
    except ValidationError as exc:
        warning(str(exc))

    tm1 = TopMulti()
    print("TopMulti", tm1)  # noqa: T201

    lm1 = LeftMulti(tsdf=1, length=2)
    print("LeftMulti", lm1)  # noqa: T201

    try:
        rm1 = RightMulti(constituents=(1, 2))
        print("RightMulti", rm1)  # noqa: T201
    except ValidationError as exc:
        warning(str(exc))
