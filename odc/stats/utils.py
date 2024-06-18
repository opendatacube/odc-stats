import toolz
from typing import Dict, Tuple, List, Any, Callable, Optional
from collections import namedtuple, defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .model import DateTimeRange, odc_uuid
from datacube.storage import measurement_paths
from datacube.model import Dataset, DatasetType
from datacube.index.eo3 import prep_eo3


CompressedDataset = namedtuple("CompressedDataset", ["id", "time"])
Cell = Any


def _bin_generic(
    dss: List[CompressedDataset], bins: List[DateTimeRange]
) -> Dict[str, List[CompressedDataset]]:
    """
    Dumb O(NM) implementation, N number of dataset, M number of bins.

    For every bin find all datasets that fall in there, and if not empty keep that bin.
    """
    out: Dict[str, List[CompressedDataset]] = {}
    for b in bins:
        _dss = [ds for ds in dss if ds.time in b]
        if len(_dss) > 0:
            out[b.short] = _dss

    return out


def bin_generic(
    cells: Dict[Tuple[int, int], Cell], bins: List[DateTimeRange]
) -> Dict[Tuple[str, int, int], List[CompressedDataset]]:
    tasks: Dict[Tuple[str, int, int], List[CompressedDataset]] = {}
    for tidx, cell in cells.items():
        _bins = _bin_generic(cell.dss, bins)
        for t, dss in _bins.items():
            tasks[(t,) + tidx] = dss

    return tasks


def bin_seasonal(
    cells: Dict[Tuple[int, int], Cell],
    months: int,
    anchor: int,
    extract_single_season=False,
) -> Dict[Tuple[str, int, int], List[CompressedDataset]]:
    # mk_single_season_rules is different from mk_season_rules
    # because the mk_season_rules will split the whole year to 2/3/4 seasons
    # but mk_single_season_rules only extract a single season from the whole year
    if extract_single_season:
        binner = season_binner(mk_single_season_rules(months, anchor))
    else:
        binner = season_binner(mk_season_rules(months, anchor))

    tasks = {}
    for tidx, cell in cells.items():
        # This is a great pylint warning, but doesn't apply here because we
        # only call the lambda from inside each iteration of the loop
        # pylint:disable=cell-var-from-loop
        utc_offset = cell.utc_offset
        grouped = toolz.groupby(lambda ds: binner(ds.time + utc_offset), cell.dss)

        for temporal_k, dss in grouped.items():
            if temporal_k != "":
                tasks[(temporal_k,) + tidx] = dss

    return tasks


def bin_rolling_seasonal(
    cells: Dict[Tuple[int, int], Cell],
    temporal_range,
    months: int,
    interval: int,
) -> Dict[Tuple[str, int, int], List[CompressedDataset]]:
    binner = rolling_season_binner(
        mk_rolling_season_rules(temporal_range, months, interval)
    )

    tasks = {}
    for tidx, cell in cells.items():
        # This is a great pylint warning, but doesn't apply here because we
        # only call the lambda from inside each iteration of the loop
        # pylint:disable=cell-var-from-loop
        utc_offset = cell.utc_offset
        _grouped = toolz.groupby(lambda ds: binner(ds.time + utc_offset), cell.dss)

        grouped = defaultdict(list)
        for key, value in _grouped.items():
            for k in key:
                grouped[k].extend(value)

        for temporal_k, dss in grouped.items():
            if temporal_k != "":
                tasks[(temporal_k,) + tidx] = dss

    return tasks


def bin_full_history(
    cells: Dict[Tuple[int, int], Cell], start: datetime, end: datetime
) -> Dict[Tuple[str, int, int], List[CompressedDataset]]:
    duration = end.year - start.year + 1
    temporal_key = (f"{start.year}--P{duration}Y",)
    return {temporal_key + k: cell.dss for k, cell in cells.items()}


def bin_annual(
    cells: Dict[Tuple[int, int], Cell]
) -> Dict[Tuple[str, int, int], List[CompressedDataset]]:
    """
    Annual binning
    :param cells: (x,y) -> Cell(dss: List[CompressedDataset], geobox: GeoBox, idx: Tuple[int, int])
    """
    tasks = {}
    for tidx, cell in cells.items():
        # This is a great pylint warning, but doesn't apply here because we
        # only call the lambda from inside each iteration of the loop
        # pylint:disable=cell-var-from-loop
        utc_offset = cell.utc_offset
        grouped = toolz.groupby(lambda ds: (ds.time + utc_offset).year, cell.dss)

        for year, dss in grouped.items():
            temporal_k = (f"{year}--P1Y",)
            tasks[temporal_k + tidx] = dss

    return tasks


def mk_single_season_rules(months: int, anchor: int) -> Dict[int, str]:
    """
    Construct rules for a each year single season summary
    :param months: Length of season in months can be one of [1, 12]
    :param anchor: Start month of the season [1, 12]
    """
    assert 1 <= months <= 12
    assert 1 <= anchor <= 12

    rules: Dict[int, str] = {}

    start_month = anchor

    # not loop as mk_season_rules(). because it only return
    # single season task cache each run
    for m in range(start_month, start_month + months):
        if m > 12:
            m = m - 12
        if months == 12:
            rules[m] = f"{start_month:02d}--P1Y"
        else:
            rules[m] = f"{start_month:02d}--P{months:d}M"

    return rules


def mk_season_rules(months: int, anchor: int) -> Dict[int, str]:
    """
    Construct rules for a regular seasons
    :param months: Length of season in months can be one of (1,2,3,4,6,12)
    :param anchor: Start month of one of the seasons [1, 12]
    """
    assert months in (1, 2, 3, 4, 6, 12)
    assert 1 <= anchor <= 12

    rules: Dict[int, str] = {}
    for i in range(12 // months):
        start_month = anchor + i * months
        if start_month > 12:
            start_month -= 12

        for m in range(start_month, start_month + months):
            if m > 12:
                m = m - 12
            if months == 12:
                rules[m] = f"{start_month:02d}--P1Y"
            else:
                rules[m] = f"{start_month:02d}--P{months:d}M"

    return rules


def mk_rolling_season_rules(temporal_range, months, interval):
    """
    Construct rules for rolling seasons
    :param temporal_range: Time range for which datasets have been loaded.
    :param months: Length of a single season in months can be one of [1, 12]
    :param interval: Length in months between the start months for 2 consecutive seasons.
    """
    assert 1 <= months <= 12
    assert 0 < interval < months

    season_start_interval = relativedelta(months=interval)

    start_date = temporal_range.start
    end_date = temporal_range.end

    rules = {}
    season_start = start_date
    while (
        DateTimeRange(f'{season_start.strftime("%Y-%m-%d")}--P{months}M').end
        <= end_date
    ):
        rules[f'{season_start.strftime("%Y-%m")}--P{months}M'] = DateTimeRange(
            f'{season_start.strftime("%Y-%m-%d")}--P{months}M'
        )
        season_start += season_start_interval

    return rules


def season_binner(rules: Dict[int, str]) -> Callable[[datetime], str]:
    """
    Construct mapping from datetime to a string in the form like 2010-06--P3M

    :param rules: Is a mapping from month (1-Jan, 2-Feb) to a string in the
                  form "{month:int}--P{N:int}M", where ``month`` is a starting
                  month of the season and ``N`` is a duration of the season in
                  months.
    """
    _rules: Dict[int, Tuple[str, int]] = {}

    for month in range(1, 12 + 1):
        season = rules.get(month, "")
        if season == "":
            _rules[month] = ("", 0)
        else:
            start_month = int(season.split("--")[0])
            _rules[month] = (season, 0 if start_month <= month else -1)

    def label(dt: datetime) -> str:
        season, yoffset = _rules[dt.month]
        if season == "":
            return ""
        y = dt.year + yoffset
        return f"{y}-{season}"

    return label


def rolling_season_binner(rules: Dict[int, str]) -> Callable[[datetime], list]:
    """
    Construct mapping from datetime to a string in the form like 2010-06--P3M

    :param rules: Is a mapping from month (1-Jan, 2-Feb) to a string in the
                  form "{month:int}--P{N:int}M", where ``month`` is a starting
                  month of the season and ``N`` is a duration of the season in
                  months.

    """

    def label(dt: datetime) -> list:
        labels = []
        for label, label_date_range in rules.items():
            if dt in label_date_range:
                labels.append(label)

        return tuple(labels)

    return label


def dedup_s2_datasets(dss):
    """
    De-duplicate Sentinel 2 datasets. Datasets that share same timestamp and
    region code are considered to be duplicates.

    - Sort Datasets by ``(time, region code, label)``
    - Find groups of dataset that share common ``(time, region_code)``
    - Out of duplicate groups pick one with the most recent timestamp in the label (processing time)

    Above, ``label`` is something like this:
    ``S2B_MSIL2A_20190507T093039_N0212_R136_T32NPF_20190507T122231``

    The two timestamps are "capture time" and "processing time".

    :returns: Two list of Datasets, first one contains "freshest" datasets and
              has no duplicates, and the second one contains less fresh duplicates.
    """
    dss = sorted(
        dss,
        key=lambda ds: (
            ds.center_time,
            ds.metadata.region_code,
            ds.metadata_doc["label"],
        ),
    )
    out = []
    skipped = []

    for chunk in toolz.partitionby(
        lambda ds: (ds.center_time, ds.metadata.region_code), dss
    ):
        out.append(chunk[-1])
        skipped.extend(chunk[:-1])
    return out, skipped


def fuse_products(*ds_types) -> DatasetType:
    """
    Fuses two products. This function requires access to a Datacube to access the metadata type.

    Fusing two products requires that:
      - both metadata types are eo3
      - there are no conflicting band names
      - the file formats are identical
    """

    if len(ds_types) < 2:
        raise ValueError("Number of products to be fused must be >= 2")

    def_s = [s.definition for s in ds_types]
    fused_def = {}

    if not all(d["metadata_type"] == "eo3" for d in def_s):
        raise ValueError("metadata_type must be eo3")

    fused_def["metadata_type"] = "eo3"
    measurements = []
    for d in def_s:
        measurements += [m["name"] for m in d["measurements"]]

    if len(measurements) > len(set(measurements)):
        raise ValueError("Measurements are overlapping, they should be different")

    file_format = None
    try:
        file_format = def_s[0]["metadata"]["properties"]["odc:file_format"]
        if not all(
            file_format == d["metadata"]["properties"]["odc:file_format"] for d in def_s
        ):
            raise ValueError("odc:file_format was different between scenes")
    except KeyError:
        # odc:file_format didn't exist, but it's not required, so we're good
        pass

    name = "fused__" + "__".join([d["name"] for d in def_s])

    fused_def["name"] = name
    fused_def["metadata"] = {"product": {"name": name}}

    if file_format is not None:
        fused_def["metadata"]["odc:file_format"] = file_format

    fused_def["description"] = "Fused products: " + ",".join(
        [def_s[i].get("description", f"non_indexed_{i}") for i in range(len(def_s))]
    )
    fused_def["measurements"] = []
    for d in def_s:
        fused_def["measurements"] += d["measurements"]

    return DatasetType(ds_types[0].metadata_type, fused_def)


def fuse_ds(
    *dss,
    product: Optional[DatasetType] = None,
) -> Dataset:
    """
    This function fuses two datasets. It requires that:
      - the products are fusable
      - grids with the same name are identical
      - labels are in the format 'product_suffix' with identical suffixes
      - CRSs' are identical
      - datetimes are identical
      - $schemas are identical
    """
    # pylint:disable=too-many-locals,too-many-branches,consider-using-enumerate
    if len(dss) < 2:
        raise ValueError("Number of products to be fused must be >= 2")

    doc_s = [ds.metadata_doc for ds in dss]

    if product is None:
        product = fuse_products(*[ds.type for ds in dss])

    fused_doc = {
        "id": str(odc_uuid(product.name, "0.0.0", sources=[d["id"] for d in doc_s])),
        "lineage": {"source_datasets": [d["id"] for d in doc_s]},
    }

    # check that all grids with the same name are identical
    common_grids = set(doc_s[0]["grids"].keys())
    for d in doc_s[1:]:
        common_grids &= set(d["grids"].keys())
    match_grid = True
    for g in common_grids:
        # not sure why z-affine was omitted in odc.stac
        # special treatment for transform until we know
        for k, v in doc_s[0]["grids"][g].items():
            if k == "transform":
                match_grid &= all(
                    list(d["grids"][g][k])[:6] == list(v)[:6] for d in doc_s[1:]
                )
            else:
                match_grid &= all(list(d["grids"][g][k]) == list(v) for d in doc_s[1:])
    if not match_grid:
        raise ValueError("Grids are not all the same")

    # TODO: handle the case that grids have conflicts in a seperate function
    fused_doc["grids"] = {**doc_s[0]["grids"]}
    for d in doc_s[1:]:
        for k, v in d["grids"].items():
            fused_doc["grids"].setdefault(k, v)

    # This is to match grid, time and maturity
    label_title_doc_s = [
        d.get("label", toolz.get_in(["properties", "title"], d)) for d in doc_s
    ]
    if any(lt is None for lt in label_title_doc_s):
        raise ValueError("No label or title field found found")

    for i in range(len(label_title_doc_s)):
        label_title_doc_s[i] = label_title_doc_s[i].replace(dss[i].product.name, "")

    lt = label_title_doc_s[0]
    if any(t != lt for t in label_title_doc_s[1:]):
        raise ValueError(f"Label/Title field must be same for all {label_title_doc_s}")

    fused_doc["label"] = f"{product.name}{lt}"

    equal_keys = ["$schema", "crs"]
    for key in equal_keys:
        v = doc_s[0][key]
        if any(d[key].casefold() != v.casefold() for d in doc_s[1:]):
            raise ValueError(f"{key} is not the same")
        fused_doc[key] = v

    # datetime is the only mandatory property
    dt = doc_s[0]["properties"]["datetime"].replace("Z", "+00:00")
    if any(d["properties"]["datetime"].replace("Z", "+00:00") != dt for d in doc_s[1:]):
        raise ValueError("Datetimes are not the same")
    fused_doc["properties"] = {"datetime": dt}

    # copy over all identical properties
    match_keys = set(doc_s[0]["properties"].keys())
    for d in doc_s[1:]:
        match_keys &= set(d["properties"].keys())

    match_keys -= {"datetime"}
    for k in match_keys:
        v = doc_s[0]["properties"][k]
        if all(d["properties"][k] == v for d in doc_s[1:]):
            fused_doc["properties"][k] = v

    fused_doc["measurements"] = {}
    for d in doc_s:
        fused_doc["measurements"].update({**d["measurements"]})

    for ds in dss:
        for key, path in {**measurement_paths(ds)}.items():
            fused_doc["measurements"][key]["path"] = path

    fused_ds = Dataset(product, prep_eo3(fused_doc), uris=[""])
    fused_doc["properties"]["fused"] = "True"
    return fused_ds
