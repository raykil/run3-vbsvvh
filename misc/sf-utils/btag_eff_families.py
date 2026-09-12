"""Canonical b-tag efficiency family/channel configuration."""

from pathlib import Path

import yaml


CONFIG_DIR = (Path(__file__).resolve().parents[2] / "preselection" / "corrections" /
              "scalefactors" / "btagging")
CONFIG_PATH = CONFIG_DIR / "btag_eff_families.yaml"


def config_path_for_year(year):
    """Return the one canonical family/merge configuration for every era."""
    return CONFIG_PATH


def _nonempty_string_list(value, context):
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{context} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{context} contains duplicate entries")


def _validate_final_groups(groups, expected_members, kind):
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"final_merges.{kind} must be a non-empty mapping")
    occurrences = {}
    for group, members in groups.items():
        if not isinstance(group, str) or not group:
            raise ValueError(f"final_merges.{kind} has an invalid group name")
        _nonempty_string_list(members, f"final_merges.{kind}.{group}")
        for member in members:
            occurrences.setdefault(member, []).append(group)
    missing = set(expected_members) - set(occurrences)
    unknown = set(occurrences) - set(expected_members)
    duplicate = {member: names for member, names in occurrences.items() if len(names) != 1}
    if missing or unknown or duplicate:
        raise ValueError(
            f"final_merges.{kind} must assign every source exactly once; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}, duplicates={duplicate}")


def validate_config(config):
    """Validate the narrow YAML schema consumed by both Python and C++."""
    expected_top = {"preliminary_families", "final_merges", "excluded_source_channels"}
    if not isinstance(config, dict) or set(config) != expected_top:
        raise ValueError(f"b-tag family YAML must contain exactly {sorted(expected_top)}")
    preliminary = config["preliminary_families"]
    if not isinstance(preliminary, dict) or not preliminary:
        raise ValueError("preliminary_families must be a non-empty mapping")
    for family, needles in preliminary.items():
        if not isinstance(family, str) or not family:
            raise ValueError("preliminary_families has an invalid family name")
        _nonempty_string_list(needles, f"preliminary_families.{family}")

    final = config["final_merges"]
    if not isinstance(final, dict) or set(final) != {"samples", "channels"}:
        raise ValueError("final_merges must contain exactly samples and channels")
    _validate_final_groups(final["samples"], preliminary, "samples")
    retained_channels = [channel for members in final["channels"].values() for channel in members]
    _validate_final_groups(final["channels"], retained_channels, "channels")

    excluded = config["excluded_source_channels"]
    _nonempty_string_list(excluded, "excluded_source_channels")
    overlap = set(retained_channels) & set(excluded)
    if overlap:
        raise ValueError(f"excluded_source_channels also appear in final_merges.channels: {sorted(overlap)}")
    return config


def _validate_canonical_layout(text):
    """Reject YAML constructs outside the fixed layout consumed by C++."""
    section = final_kind = current_group = None
    top_level = {"preliminary_families:", "final_merges:", "excluded_source_channels:"}
    for raw_line in text.splitlines():
        if "\t" in raw_line:
            raise ValueError("canonical b-tag family YAML does not allow tabs")
        line = raw_line.split("#", 1)[0]
        content = line.strip()
        if not content:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            if content not in top_level:
                raise ValueError(f"unknown canonical YAML top-level entry {content!r}")
            section, final_kind, current_group = content[:-1], None, None
        elif section == "preliminary_families":
            if indent == 2 and content.endswith(":"):
                current_group = content[:-1]
            elif indent == 4 and content.startswith("- ") and current_group:
                pass
            else:
                raise ValueError("invalid preliminary_families YAML layout")
        elif section == "excluded_source_channels":
            if indent != 2 or not content.startswith("- "):
                raise ValueError("invalid excluded_source_channels YAML layout")
        elif section == "final_merges":
            if indent == 2 and content in {"samples:", "channels:"}:
                final_kind, current_group = content[:-1], None
            elif indent == 4 and content.endswith(":") and final_kind:
                current_group = content[:-1]
            elif indent == 6 and content.startswith("- ") and current_group:
                pass
            else:
                raise ValueError("invalid final_merges YAML layout")
        else:
            raise ValueError("content outside the canonical b-tag family YAML sections")


def load_config(year, path=None):
    path = config_path_for_year(year) if path is None else path
    text = Path(path).read_text()
    _validate_canonical_layout(text)
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or set(raw) != {"preliminary_families", "final_merges", "excluded_source_channels"}:
        raise ValueError("canonical b-tag family YAML has an invalid top-level schema")
    return validate_config(raw)


def sample_family(sample, config=None, year=None):
    """Return the first preliminary family match; YAML order resolves overlaps."""
    config = load_config(year) if config is None else config
    for family, needles in config["preliminary_families"].items():
        if any(needle in sample for needle in needles):
            return family
    raise ValueError(f"No preliminary b-tag efficiency family is configured for sample {sample!r}")


def final_group(kind, preliminary_name, config=None, year=None):
    config = load_config(year) if config is None else config
    matches = [name for name, members in config["final_merges"][kind].items()
               if preliminary_name in members]
    if len(matches) != 1:
        raise ValueError(f"{preliminary_name!r} must occur in exactly one final {kind} group; found {matches}")
    return matches[0]


def final_sample_family(sample, config=None, year=None):
    config = load_config(year) if config is None else config
    return final_group("samples", sample_family(sample, config), config)


def final_channel(channel, config=None, year=None):
    return final_group("channels", channel, config, year)


def retained_source_channels(config=None, year=None):
    config = load_config(year) if config is None else config
    return tuple(channel for members in config["final_merges"]["channels"].values() for channel in members)


def excluded_source_channels(config=None, year=None):
    config = load_config(year) if config is None else config
    return tuple(config["excluded_source_channels"])

def final_runtime_keys(year, channel, sample, config=None):
    """The strict new-schema keys required by the C++ runtime lookup."""
    config = load_config(year) if config is None else config
    return f"btag_{year}_{final_channel(channel, config)}", final_sample_family(sample, config)
