#!/usr/bin/env python3


from cocomo import ComponentType


def main() -> None:
    types = ComponentType.read_list("component_types", dir=".")
    types["hexamer"].writeout("sasa", "hexamer.surface")
    types["hexamer"].writeout("enm", "hexamer.enmpairs")
    types["pentamer"].writeout("sasa", "pentamer.surface")
    types["pentamer"].writeout("enm", "pentamer.enmpairs")
    types["trimer"].writeout("sasa", "trimer.surface")
    types["trimer"].writeout("enm", "trimer.enmpairs")


if __name__ == "__main__":
    main()
