import supgml


def test_base_package_has_no_optional_import_side_effects():
    assert supgml.__version__ == "0.1.0"
