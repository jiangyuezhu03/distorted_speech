from typing import List

import pytest

from docstring_parser.rest_attr_parser import parse_attributes, Attribute


@pytest.mark.parametrize(
    'docstring, expected_attributes, expected_lines_with_attributes',
    [
        (
            '',
            [],
            [],
        ),
        (
            """
            My Class

            :param name: My name
            :type name: str
            """,
            [],
            [],
        ),
        (
            """
            My Class

            .. attribute :: attr_1
                :type: str

            .. attribute :: attr_2
                :type: bool

                Attr 2

            .. attribute :: attr_3

                Attr 3

            .. attribute :: attr_4
            .. attribute :: attr_5
            .. attribute :: attr_6
                :type: dict | list

                !

            :param name: My name
            :type name: str
            """,
            [
                Attribute(name='attr_1', type='str', description=None),
                Attribute(name='attr_2', type='bool', description='Attr 2'),
                Attribute(name='attr_3', type=None, description='Attr 3'),
                Attribute(name='attr_4', type=None, description=None),
                Attribute(name='attr_5', type=None, description=None),
                Attribute(name='attr_6', type='dict | list', description='!'),
            ],
            [3, 4, 6, 7, 9, 11, 13, 15, 16, 17, 18, 20],
        ),
        (
            """
            My Class

            .. attribute :: attr_1
                :type: str

            .. attribute :: attr_2
                :type: bool

                Attr 2

            :param bar: A param called "bar"
            :type name: float

            .. attribute :: attr_3

                Attr 3
            :param goo: A param called "goo"
            :type name: bool

            .. attribute :: attr_4
            .. attribute :: attr_5
            :param foo: A param called "foo"
            :type name: float
            .. attribute :: attr_6
                :type: dict | list

                !

            :param name: My name
            :type name: str
            """,
            [
                Attribute(name='attr_1', type='str', description=None),
                Attribute(name='attr_2', type='bool', description='Attr 2'),
                Attribute(name='attr_3', type=None, description='Attr 3'),
                Attribute(name='attr_4', type=None, description=None),
                Attribute(name='attr_5', type=None, description=None),
                Attribute(name='attr_6', type='dict | list', description='!'),
            ],
            [3, 4, 6, 7, 9, 14, 16, 20, 21, 24, 25, 27],
        ),
    ],
)
def test_parser_attributes(
        docstring: str,
        expected_attributes: List[Attribute],
        expected_lines_with_attributes: List[str],
) -> None:
    attributes, lines_with_attributes = parse_attributes(docstring)
    assert attributes == expected_attributes
    assert lines_with_attributes == expected_lines_with_attributes
