"""Dataclasses for the ePCR categories, after the JSON templates in templates/.

Each vital sign and each demographic is its own dataclass (BloodPressure, GCS, Age,
Sex, ...), holding only its own fields and checking its own rules in __post_init__,
so an invalid one can't be built at all. An item's "entity" picks its class
(VITAL_SIGNS, DEMOGRAPHICS); from_dict() turns the model's JSON into the class,
converting what it can ("111" -> 111, "M" -> "male") and raising ValidationError
otherwise. Demographics is one record with a field per demographic, and is
complete once every field is known. Medical history items all have the same
fields, so they share one class, MedicalHistory, as the mechanism of injury has
MechanismOfInjury.

The module does two jobs for the extraction service:
  1. validate(category, reply) checks a category of the model's reply and gives
     it back in the templates' format, keeping the valid items and reporting the
     rest.
  2. is_complete(category, state) tells when demographics or mechanism_of_injury
     is done, so its extraction call can be skipped for the rest of the session.

The JSON is as in the templates: {"entity", "standard_name", <fields>, "quotes"},
each quote a {"text", "time"} object, and every quote has its time. The model only
returns quote text (a quote may be a plain string), so parse() and validate() take
time_of(text), which gives a quote's time, or None when it can't be found in the
transcript. Such a quote is dropped and reported, and an item left with no quote is
invalid.

    state, errors = validate("vital_signs", reply["vital_signs"], time_of)
    if is_complete("demographics", state["demographics"]): ...skip the call
"""

import re
from dataclasses import MISSING, dataclass, fields
from typing import Any, Callable, ClassVar, Optional, Union, get_args, get_type_hints

Number = Union[int, float]


class ValidationError(ValueError):
    """An item of the model's reply doesn't fit its dataclass."""


class QuoteNotFound(ValidationError):
    """A quote has no time and can't be found in the transcript; the quote is dropped,
    not its item."""


TimeOf = Callable[[str], Optional[float]]


# --- reading the model's JSON ----------------------------------------------------

# A number, optionally followed by its unit: "111", "37.5", "80 kg", "35.2°C", "18 cmH2O".
# A unit starts with a letter, %, ° or µ, so "140/89" is not 140 in "/89".
_NUMBER = re.compile(r"\s*([-+]?\d+(?:\.\d+)?)\s*((?:[^\W\d_]|[%°µ]).*?)?\s*")


def _text(d: dict, key: str, required: bool = True) -> Optional[str]:
    v = d.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise ValidationError(f"{key!r} is missing")
        return None
    if not isinstance(v, str):
        raise ValidationError(f"{key!r} must be text, got {v!r}")
    return v.strip()


def _number(d: dict, key: str, required: bool = True) -> tuple[Optional[Number], Optional[str]]:
    """(number, unit) of d[key]: a number, or text that is a number optionally followed by its
    unit ("111" -> (111, None), "80 kg" -> (80, "kg")); bools aren't numbers."""
    v = d.get(key)
    if v is None or v == "":
        if required:
            raise ValidationError(f"{key!r} is missing")
        return None, None
    unit = None
    if isinstance(v, str) and (m := _NUMBER.fullmatch(v)):
        v = float(m[1]) if "." in m[1] else int(m[1])
        unit = m[2]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValidationError(f"{key!r} must be a number, got {v!r}")
    return v, unit


def _quotes(d: dict, time_of: Optional[TimeOf]) -> list["Quote"]:
    """The item's quotes, each timed; a quote that can't be found is left out (the item's
    own check then fails if none is left)."""
    v = d.get("quotes")
    if not isinstance(v, list):
        raise ValidationError(f"'quotes' must be a list, got {v!r}")
    quotes = []
    for q in v:
        try:
            quotes.append(Quote.from_dict(q, time_of))
        except QuoteNotFound:
            continue
    return quotes


def _dict(d: Any) -> dict:
    if not isinstance(d, dict):
        raise ValidationError(f"item must be an object, got {d!r}")
    return d


def _entity_name(d: dict) -> str:
    """An item's entity, normalized: "Heart rate" -> "heart_rate"."""
    return re.sub(r"[\s-]+", "_", _text(d, "entity").lower())


def _from_dict(cls, d: Any, time_of: Optional[TimeOf] = None):
    """Builds dataclass cls from the model's JSON, reading each field by its type: quotes,
    numbers and text. A vital sign's or demographic's entity is its class (ENTITY), not a
    field; keys cls doesn't have are ignored. Quotes are timed with time_of (see Quote).
    __post_init__ then checks the values.

    A number given with its unit ("80 kg") is split: the unit goes in the number's own unit
    field, <field>_unit (a medication's dose_unit, rate_unit), or else in the class's unit
    field, unless that unit is given and is the same. Two different units, or a unit on a
    number with no unit field (GCS), is an error.
    """
    d = _dict(d)
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unit_fields = {n for n in names if n == "unit" or n.endswith("_unit")}
    units = {n: _text(d, n, False) for n in unit_fields}  # set below, from the JSON or the numbers
    kwargs = {}
    for f in fields(cls):
        required = f.default is MISSING and f.default_factory is MISSING
        types = set(get_args(hints[f.name])) - {type(None)} or {hints[f.name]}
        if f.name == "quotes":
            kwargs["quotes"] = _quotes(d, time_of)
            continue
        if f.name in unit_fields:
            continue
        if types <= {int, float}:
            v, stated = _number(d, f.name, required)
            if stated is not None:
                target = f"{f.name}_unit" if f"{f.name}_unit" in names else "unit" if "unit" in names else None
                if target is None:
                    raise ValidationError(f"{f.name!r} must be a number, got {d[f.name]!r}")
                if units[target] is not None and stated.lower() != units[target].lower():
                    raise ValidationError(f"{f.name!r} is in {stated!r}, but {target} is {units[target]!r}")
                units[target] = units[target] or stated
        else:
            v = _text(d, f.name, required)
        if v is not None:
            kwargs[f.name] = v
    kwargs.update({n: u for n, u in units.items() if u is not None})
    return cls(**kwargs)


# --- writing the templates' JSON ----------------------------------------------------

def _to_dict(obj) -> dict:
    """A dataclass as in the templates: its entity, then its fields in order; fields that are
    None are left out."""
    out = {"entity": obj.ENTITY} if hasattr(obj, "ENTITY") else {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if f.name == "quotes":
            v = [{"text": q.text, "time": q.time} for q in v]
        if v is not None:
            out[f.name] = v
    return out


# --- checks, for __post_init__ ---------------------------------------------------------

def _check_quotes(obj) -> None:
    if not obj.quotes:
        raise ValidationError("'quotes' is empty")


def _check_range(obj, name: str, low: Number = None, high: Number = None, whole: bool = False) -> None:
    """Field name, when set, lies in [low, high] (and is a whole number if whole; it is then
    stored as an int)."""
    v = getattr(obj, name)
    if v is None:
        return
    if whole:
        if v != int(v):
            raise ValidationError(f"{name} must be a whole number, got {v!r}")
        setattr(obj, name, v := int(v))
    if (low is not None and v < low) or (high is not None and v > high):
        bounds = f"from {low} to {high}" if high is not None else f"at least {low}"
        raise ValidationError(f"{getattr(obj, 'ENTITY', name)} {name} must be {bounds}, got {v!r}")


def _default_name(obj) -> None:
    obj.standard_name = obj.standard_name or obj.STANDARD_NAME


# --- quotes ------------------------------------------------------------------------------

@dataclass
class Quote:
    """An utterance, word for word, and when it was said (seconds into the session)."""

    text: str
    time: float

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("quote text is missing")
        if isinstance(self.time, bool) or not isinstance(self.time, (int, float)):
            raise ValidationError(f"quote time must be a number, got {self.time!r}")
        self.time = float(self.time)

    @classmethod
    def from_dict(cls, d: Union[str, dict], time_of: Optional[TimeOf] = None) -> "Quote":
        """A quote from {"text", "time"}, or from its text alone (a string, or no "time"),
        timed by time_of. Raises QuoteNotFound when time_of can't find it, and
        ValidationError when there is no time_of to ask."""
        if isinstance(d, str):
            d = {"text": d}
        d = _dict(d)
        text, t = _text(d, "text"), d.get("time")
        if t is None:
            if time_of is None:
                raise ValidationError(f"quote {text!r} has no time")
            t = time_of(text)
            if t is None:
                raise QuoteNotFound(f"quote not found in the transcript: {text!r}")
        return cls(text=text, time=t)


# --- vital signs: one class per vital sign -------------------------------------------------
# kw_only, so the required quotes can come last, after the optional fields, as in the templates.

@dataclass(kw_only=True)
class BloodPressure:
    ENTITY: ClassVar[str] = "blood_pressure"
    STANDARD_NAME: ClassVar[str] = "blood pressure"

    standard_name: Optional[str] = None
    systolic: Number
    diastolic: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "diastolic", 1)
        if self.systolic <= self.diastolic:
            raise ValidationError(f"systolic {self.systolic} is not above diastolic {self.diastolic}")

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "BloodPressure":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class MeanArterialPressure:
    ENTITY: ClassVar[str] = "map"
    STANDARD_NAME: ClassVar[str] = "mean arterial pressure"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 1)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "MeanArterialPressure":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class HeartRate:
    ENTITY: ClassVar[str] = "heart_rate"
    STANDARD_NAME: ClassVar[str] = "heart rate"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "HeartRate":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Saturation:
    ENTITY: ClassVar[str] = "saturation"
    STANDARD_NAME: ClassVar[str] = "oxygen saturation"

    standard_name: Optional[str] = None
    value: Number  # percent
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0, 100)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Saturation":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class RespiratoryRate:
    ENTITY: ClassVar[str] = "respiratory_rate"
    STANDARD_NAME: ClassVar[str] = "respiratory rate"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "RespiratoryRate":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class EndTidalCO2:
    ENTITY: ClassVar[str] = "end_tidal_co2"
    STANDARD_NAME: ClassVar[str] = "end-tidal carbon dioxide"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "EndTidalCO2":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class PeakPressure:
    ENTITY: ClassVar[str] = "peak_pressure"
    STANDARD_NAME: ClassVar[str] = "peak pressure"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "PeakPressure":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Temperature:
    ENTITY: ClassVar[str] = "temperature"
    STANDARD_NAME: ClassVar[str] = "body temperature"

    standard_name: Optional[str] = None
    value: Number  # °C or °F, as stated
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 1)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Temperature":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class GCS:
    """The total (value, 3-15), and the eye (1-4), verbal (1-5) and motor (1-6) scores when
    stated; when all three are, they add up to the total."""

    ENTITY: ClassVar[str] = "gcs"
    STANDARD_NAME: ClassVar[str] = "Glasgow Coma Scale"

    standard_name: Optional[str] = None
    value: int
    eye: Optional[int] = None
    verbal: Optional[int] = None
    motor: Optional[int] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 3, 15, whole=True)
        _check_range(self, "eye", 1, 4, whole=True)
        _check_range(self, "verbal", 1, 5, whole=True)
        _check_range(self, "motor", 1, 6, whole=True)
        parts = (self.eye, self.verbal, self.motor)
        if None not in parts and sum(parts) != self.value:
            raise ValidationError(f"GCS parts add up to {sum(parts)}, not {self.value}")

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "GCS":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Glucose:
    ENTITY: ClassVar[str] = "glucose"
    STANDARD_NAME: ClassVar[str] = "blood glucose"

    standard_name: Optional[str] = None
    value: Number  # mmol/L or mg/dL, as stated
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Glucose":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


VitalSign = Union[BloodPressure, MeanArterialPressure, HeartRate, Saturation, RespiratoryRate,
                  EndTidalCO2, PeakPressure, Temperature, GCS, Glucose]
VITAL_SIGNS: dict[str, type] = {cls.ENTITY: cls for cls in get_args(VitalSign)}


# --- demographics: one class per demographic, and one record of them -------------------------

@dataclass(kw_only=True)
class Age:
    ENTITY: ClassVar[str] = "age"
    STANDARD_NAME: ClassVar[str] = "age"

    standard_name: Optional[str] = None
    value: Number  # years, unless unit says otherwise
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0, 130)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Age":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Sex:
    ENTITY: ClassVar[str] = "sex"
    STANDARD_NAME: ClassVar[str] = "sex"
    SEXES: ClassVar[dict[str, str]] = {"male": "male", "m": "male", "man": "male",
                                       "female": "female", "f": "female", "woman": "female"}

    standard_name: Optional[str] = None
    value: str  # "male" or "female"
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        sex = self.SEXES.get(str(self.value).strip().lower())
        if sex is None:
            raise ValidationError(f"sex must be male or female, got {self.value!r}")
        self.value = sex

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Sex":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Weight:
    ENTITY: ClassVar[str] = "weight"
    STANDARD_NAME: ClassVar[str] = "weight"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0.1)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Weight":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class Height:
    ENTITY: ClassVar[str] = "height"
    STANDARD_NAME: ClassVar[str] = "height"

    standard_name: Optional[str] = None
    value: Number
    unit: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        _default_name(self)
        _check_quotes(self)
        _check_range(self, "value", 0.1)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Height":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass
class Demographics:
    """The patient's demographics, one field per demographic; complete once every field is known.

    In JSON it is the templates' list of items, in field order; a demographic stated twice
    keeps the later statement.
    """

    age: Optional[Age] = None
    sex: Optional[Sex] = None
    weight: Optional[Weight] = None
    height: Optional[Height] = None

    @property
    def complete(self) -> bool:
        """Every demographic is known, so the extraction call can stop."""
        return not self.missing()

    def missing(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]

    @classmethod
    def from_list(cls, reply: Any, time_of: Optional[TimeOf] = None) -> tuple["Demographics", list[str]]:
        """The record of a list of items, and one error per item that isn't valid."""
        record, errors = cls(), []
        for i, d in enumerate(reply if isinstance(reply, list) else []):
            try:
                item = _parse_item(DEMOGRAPHICS, d, time_of)
            except ValidationError as e:
                errors.append(f"demographics[{i}]: {e}")
                continue
            setattr(record, item.ENTITY, item)
        if not isinstance(reply, list):
            errors.append(f"demographics: must be a list, got {type(reply).__name__}")
        return record, errors

    def to_list(self) -> list[dict]:
        return [v.to_dict() for f in fields(self) if (v := getattr(self, f.name)) is not None]


DEMOGRAPHICS: dict[str, type] = {cls.ENTITY: cls for cls in (Age, Sex, Weight, Height)}


# --- medical history and mechanism of injury ------------------------------------------------

@dataclass(kw_only=True)
class MedicalHistory:
    """The patient's own history, their family's, or a comorbidity; the content is in its quotes.
    The three have the same fields, so they share this class."""

    ENTITIES: ClassVar[tuple[str, ...]] = ("family_history", "patient_history", "comorbidity")

    entity: str
    standard_name: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        self.entity = re.sub(r"[\s-]+", "_", self.entity.strip().lower())
        if self.entity not in self.ENTITIES:
            raise ValidationError(f"entity {self.entity!r} is not one of {', '.join(self.ENTITIES)}")
        self.standard_name = self.standard_name or self.entity.replace("_", " ")
        _check_quotes(self)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "MedicalHistory":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class MechanismOfInjury:
    """How the injury happened, as one description."""

    entity: str
    quotes: list[Quote]

    def __post_init__(self):
        if not self.entity.strip():
            raise ValidationError("'entity' is missing")
        _check_quotes(self)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "MechanismOfInjury":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


# --- interventions ----------------------------------------------------------------------------

@dataclass(kw_only=True)
class Intervention:
    """A procedure performed on the patient: what was done (entity, e.g. "endotracheal
    intubation"), which of the CATEGORIES it is, and its details when stated (size, site,
    settings). The categories must match intervention_categories in prompts/_shared.yaml."""

    CATEGORIES: ClassVar[tuple[str, ...]] = (
        "advanced_airway", "mechanical_ventilation", "oxygen_therapy", "chest_decompression",
        "ultrasound", "cardiac_monitoring", "electrical_therapy", "vascular_access",
        "spinal_motion_restriction", "blood_product", "cardiac_arrest_care",
    )

    entity: str
    category: str
    details: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        if not self.entity.strip():
            raise ValidationError("'entity' is missing")
        self.category = re.sub(r"[\s-]+", "_", self.category.strip().lower())
        if self.category not in self.CATEGORIES:
            raise ValidationError(f"category {self.category!r} is not one of {', '.join(self.CATEGORIES)}")
        _check_quotes(self)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Intervention":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


# --- medications --------------------------------------------------------------------------------
#
# A medication is tracked as administrations: each bolus given or infusion started is one, with an
# id the extractor assigns ("ketamine-1"). The model reports actions: a new administration
# (Medication), or an update to one by its id (MedicationUpdate): details stated later, a new rate,
# or an infusion stopped. merge_medications() turns the log of actions into each administration's
# current state.

ROUTES: dict[str, tuple[str, ...]] = {
    # Each route, and the abbreviations of it that are accepted and stored as the route.
    "intravenous": ("iv",), "intraosseous": ("io",), "intramuscular": ("im",),
    "subcutaneous": ("sc", "sq", "subcut"), "intranasal": ("in",), "inhaled": ("neb", "nebulized"),
    "oral": ("po",), "sublingual": ("sl",), "buccal": (), "rectal": ("pr",), "transdermal": (),
}


def _check_route(obj) -> None:
    """obj.route, when set, as one of ROUTES ("IV push" is intravenous)."""
    if obj.route is None:
        return
    route = obj.route.strip().lower()
    first = route.split()[0] if route else ""
    canonical = next((r for r, abbr in ROUTES.items()
                      if route == r or route in abbr or first == r or first in abbr), None)
    if canonical is None:
        raise ValidationError(f"route {obj.route!r} is not one of {', '.join(ROUTES)}")
    obj.route = canonical


@dataclass(kw_only=True)
class Medication:
    """A new administration: a bolus "given" or an infusion "started", as named in the transcript
    (entity, "ket"), its generic name (standard_name, "ketamine"), and its dose, rate and route
    when stated. type follows from the action. A bolus has no rate. The routes must match
    medication_routes in prompts/_shared.yaml."""

    ACTIONS: ClassVar[dict[str, str]] = {"given": "bolus", "started": "infusion"}
    ROUTES: ClassVar[dict[str, tuple[str, ...]]] = ROUTES

    action: str
    entity: str
    standard_name: str
    type: Optional[str] = None  # set from the action
    dose: Optional[Number] = None
    dose_unit: Optional[str] = None
    rate: Optional[Number] = None
    rate_unit: Optional[str] = None
    route: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        self.action = self.action.strip().lower()
        if self.action not in self.ACTIONS:
            raise ValidationError(f"action {self.action!r} is not one of {', '.join(self.ACTIONS)}")
        self.type = self.ACTIONS[self.action]
        for name in ("entity", "standard_name"):
            if not getattr(self, name).strip():
                raise ValidationError(f"{name!r} is missing")
        _check_range(self, "dose", 0)
        _check_range(self, "rate", 0)
        if self.type == "bolus" and self.rate is not None:
            raise ValidationError("a bolus (given) has no rate; an infusion is started")
        _check_route(self)
        _check_quotes(self)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "Medication":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(kw_only=True)
class MedicationUpdate:
    """An update to an administration already extracted, named by its id (ref): "update" adds
    or corrects its details (dose, route, ...), "rate_changed" gives an infusion's new rate,
    "stopped" stops an infusion. Only what the update states is set. Whether ref exists, and
    is an infusion where that is needed, is checked by the extractor, which knows them."""

    ACTIONS: ClassVar[tuple[str, ...]] = ("update", "rate_changed", "stopped")
    DETAILS: ClassVar[tuple[str, ...]] = ("dose", "dose_unit", "rate", "rate_unit", "route")

    action: str
    ref: str
    dose: Optional[Number] = None
    dose_unit: Optional[str] = None
    rate: Optional[Number] = None
    rate_unit: Optional[str] = None
    route: Optional[str] = None
    quotes: list[Quote]

    def __post_init__(self):
        self.action = self.action.strip().lower()
        if self.action not in self.ACTIONS:
            raise ValidationError(f"action {self.action!r} is not one of {', '.join(self.ACTIONS)}")
        self.ref = self.ref.strip()
        if not self.ref:
            raise ValidationError("'ref' is missing")
        _check_range(self, "dose", 0)
        _check_range(self, "rate", 0)
        if self.action == "rate_changed" and self.rate is None:
            raise ValidationError("rate_changed has no rate")
        if self.action == "update" and all(getattr(self, f) is None for f in self.DETAILS):
            raise ValidationError("update states no detail (dose, dose_unit, rate, rate_unit, route)")
        _check_route(self)
        _check_quotes(self)

    @classmethod
    def from_dict(cls, d: Any, time_of: Optional[TimeOf] = None) -> "MedicationUpdate":
        return _from_dict(cls, d, time_of)

    def to_dict(self) -> dict:
        return _to_dict(self)


def _parse_medication(d: Any, time_of: Optional[TimeOf]):
    """A medication action as a new administration or an update, by its action."""
    action = str(_dict(d).get("action") or "").strip().lower()
    if action in Medication.ACTIONS:
        return Medication.from_dict(d, time_of)
    if action in MedicationUpdate.ACTIONS:
        return MedicationUpdate.from_dict(d, time_of)
    raise ValidationError(f"action {action!r} is not one of "
                          f"{', '.join([*Medication.ACTIONS, *MedicationUpdate.ACTIONS])}")


def merge_medications(records: list[dict]) -> list[dict]:
    """Each administration's current state, from the medications log (records of actions, each new
    administration with its "id"), in the order they were given or started:

        {"id", "type", "entity", "standard_name", "dose", "dose_unit", "rate", "rate_unit",
         "route", "status" ("given", "running" or "stopped"), "codes": {...},
         "events": [{"segment", "action", "time", <what it set>}], "quotes": [...]}

    Fields never stated are None. An update to an id that doesn't exist is skipped.
    """
    admins = {}
    for record in records:
        for it in record.get("extractions") or []:
            action = it.get("action")
            quotes = it.get("quotes") or []
            time = min((q["time"] for q in quotes if isinstance(q, dict) and q.get("time") is not None), default=None)
            stated = {f: it[f] for f in MedicationUpdate.DETAILS if it.get(f) is not None}
            event = {"segment": record.get("segment"), "action": action, "time": time, **stated}
            if action in Medication.ACTIONS:
                admin = {"id": it.get("id"), "type": Medication.ACTIONS[action],
                         "entity": it.get("entity"), "standard_name": it.get("standard_name"),
                         **{f: it.get(f) for f in MedicationUpdate.DETAILS},
                         "status": "given" if action == "given" else "running",
                         "codes": {k: it[k] for k in ("atc", "rxnorm") if k in it},
                         "events": [], "quotes": []}
                admins[admin["id"]] = admin
            else:
                admin = admins.get(it.get("ref"))
                if admin is None:
                    continue
                admin.update(stated)
                if action == "stopped":
                    admin["status"] = "stopped"
            admin["events"].append(event)
            admin["quotes"].extend(quotes)
    return list(admins.values())


# --- per category ----------------------------------------------------------------------------

# The categories this module validates, and the entities each accepts (for the prompts).
ENTITIES: dict[str, tuple[str, ...]] = {
    "demographics": tuple(DEMOGRAPHICS),
    "vital_signs": tuple(VITAL_SIGNS),
    "medical_history": MedicalHistory.ENTITIES,
}
CATEGORIES = (*ENTITIES, "mechanism_of_injury", "interventions", "medications")


def _parse_item(classes: dict[str, type], d: Any, time_of: Optional[TimeOf] = None):
    """The item d as the class its entity names, from classes."""
    name = _entity_name(_dict(d))
    if name not in classes:
        raise ValidationError(f"entity {name!r} is not one of {', '.join(classes)}")
    return classes[name].from_dict(d, time_of)


def parse(category: str, reply: Any, time_of: Optional[TimeOf] = None) -> tuple[Any, list[str]]:
    """A category of the model's reply as dataclasses, and one error per item that isn't valid
    (an invalid item is dropped, the rest are kept) and per quote time_of couldn't find
    (the quote is dropped). Quotes without a time are timed by time_of.

    demographics gives a Demographics, every other category a list.
    """
    not_found = []

    def timed(text: str) -> Optional[float]:
        t = time_of(text) if time_of is not None else None
        if t is None:
            not_found.append(f"{category}: quote not found in the transcript, dropped: {text!r}")
        return t

    obj, errors = _parse(category, reply, timed if time_of is not None else None)
    return obj, not_found + errors


def _parse(category: str, reply: Any, time_of: Optional[TimeOf]) -> tuple[Any, list[str]]:
    if category == "demographics":
        return Demographics.from_list(reply, time_of)
    if not isinstance(reply, list):
        return [], [f"{category}: must be a list, got {type(reply).__name__}"]
    items, errors = [], []
    for i, d in enumerate(reply):
        try:
            if category == "vital_signs":
                items.append(_parse_item(VITAL_SIGNS, d, time_of))
            elif category == "mechanism_of_injury":
                items.append(MechanismOfInjury.from_dict(d, time_of))
            elif category == "interventions":
                items.append(Intervention.from_dict(d, time_of))
            elif category == "medications":
                items.append(_parse_medication(d, time_of))
            else:
                items.append(MedicalHistory.from_dict(d, time_of))
        except ValidationError as e:
            errors.append(f"{category}[{i}]: {e}")
    return items, errors


def validate(category: str, reply: Any, time_of: Optional[TimeOf] = None) -> tuple[Any, list[str]]:
    """A category of the model's reply, checked, timed and back in the templates' format, and
    its errors (see parse).

    Every category is a list of its valid items (demographics: one per demographic, the
    last stated).
    """
    obj, errors = parse(category, reply, time_of)
    if category == "demographics":
        return obj.to_list(), errors
    return [it.to_dict() for it in obj], errors


def is_complete(category: str, state: Any) -> bool:
    """Whether a category's state (its JSON) is done, so its extraction call can be skipped.
    Only demographics and mechanism_of_injury ever are."""
    if category not in ("demographics", "mechanism_of_injury"):
        return False
    obj, _ = parse(category, state)
    if category == "demographics":
        return obj.complete
    return bool(obj)  # a mechanism of injury is stated
