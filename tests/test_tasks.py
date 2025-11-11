from odc.stats.tasks import TaskReader


def test_is_compatible_resolution(test_db_path):
    from odc.stats.model import product_for_plugin
    from odc.stats.plugins.gm import StatsGMS2

    product = product_for_plugin(StatsGMS2(), location="/tmp/")
    reader = TaskReader(test_db_path, product)

    assert reader.is_compatible_resolution((30, 30))
    assert not reader.is_compatible_resolution((27, 27))
    assert not reader.is_compatible_resolution((30, 27))
    assert not reader.is_compatible_resolution((27, 30))
