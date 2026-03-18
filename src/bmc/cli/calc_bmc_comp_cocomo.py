#!/usr/bin/env python3


from cocomo import ComponentType


def main() -> None:
    types = ComponentType.read_list("component_types", dir=".")

    outputs = {
        "hexamer": ["sasa", "enm"],
        "pentamer": ["sasa", "enm"],
        "trimer": ["sasa", "enm"],
    }
    suffix = {"sasa": "surface", "enm": "enmpairs"}

    for name, kinds in outputs.items():
        component = types[name]
        for kind in kinds:
            component.writeout(kind, f"{name}.{suffix[kind]}")


if __name__ == "__main__":
    main()
