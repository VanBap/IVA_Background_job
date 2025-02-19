class EnumType:
    labels = []

    @classmethod
    def to_string(cls, enum_value):
        if not isinstance(enum_value, int):
            return ''
        if 0 <= enum_value < len(cls.labels):
            return cls.labels[enum_value]
        return ''


class ActorType(EnumType):
    UNKNOWN = 0
    SYSTEM = -1
    DETECTOR = -2
    RULE_PROCESSOR = -3


class EventType(EnumType):
    UNKNOWN = 0
    FACE = 1
    HUMAN = 2
    VEHICLE = 3
    ANIMAL = 4
    LICENSE_PLATE = 5
    HEAD_HUMAN = 6

    FIRE = 100
    CROWD_HUMAN = 101
    FULL_FRAME = 102

    # others
    VEHICLE_AND_PLATE = 200

    labels = {
        FACE: 'face',
        HUMAN: 'human',
        VEHICLE: 'vehicle',
        ANIMAL: 'animal',
        LICENSE_PLATE: 'license_plate',
        HEAD_HUMAN: 'head_human',
        FIRE: 'fire',
        VEHICLE_AND_PLATE: 'vehicle_and_plate',
        CROWD_HUMAN: 'crowd_human',
        FULL_FRAME: 'full_frame',
    }


class Quality(EnumType):
    UNKNOWN = 0
    GOOD = 1
    MEDIUM = 2
    BAD = 3

    labels = ['unknown', 'good', 'medium', 'bad']


class FaceQuality(Quality):
    pass


class FaceAgeRangeOld(EnumType):
    Age_0_12 = 1
    Age_13_17 = 2
    Age_18_34 = 3
    Age_35_54 = 4
    Age_55_100 = 5

    labels = ['unknown', '0-12', '13-17', '18-34', '35-54', '55+']


class FaceAgeRange(EnumType):
    Age_0_12 = 1
    Age_13_17 = 2
    Age_18_25 = 3
    Age_26_39 = 4
    Age_40_54 = 5
    Age_55_100 = 6

    labels = ['unknown', '0-12', '13-17', '18-25', '26-39', '40-54', '55+']


class Gender(EnumType):
    FEMALE = 1
    MALE = 2

    labels = ['unknown', 'female', 'male']


# glasses, mask, hat, bag
class Boolean(EnumType):
    NO = 1
    YES = 2
    labels = ['unknown', 'no', 'yes']


class HumanAgeRange(EnumType):
    CHILD = 1
    ADULT = 2

    labels = ['unknown', 'child', 'adult']


class UpperWearType(EnumType):
    SHORT = 1
    LONG = 2

    labels = ['unknown', 'short', 'long']


class LowerWearType(EnumType):
    SHORT = 1
    LONG = 2

    labels = ['unknown', 'short', 'long']


class Color(EnumType):
    BROWN = 1
    RED = 2
    ORANGE = 3
    YELLOW = 4
    GREEN = 5
    LIME = 6
    CYAN = 7
    PURPLE = 8
    PINK = 9
    WHITE = 10
    GRAY = 11
    BLACK = 12

    labels = ['unknown', 'brown', 'red', 'orange', 'yellow', 'green', 'lime', 'cyan',
              'purple', 'pink', 'white', 'gray', 'black']


class PrecisionLevel(EnumType):
    HIGH = 1
    NORMAL = 2
    LOW = 3

    labels = ['unknown', 'high', 'normal', 'low']


class MatchingMode(EnumType):
    INCLUDE = 1
    EXCLUDE = 2

    labels = ['unknown', 'include', 'exclude']


class MatchedStatus(EnumType):
    MATCHED = 1
    UNKNOWN = 2
    NOT_MATCHED = 3

    labels = ['', 'matched', 'unknown', 'stranger']


class VehicleTypeOld(EnumType):
    UNKNOWN = 0
    MOTO = 1  # moto
    CAR = 2  # car
    FIVE_TO_SEVEN_SEATER_SHUTTLE_BUS = 3  # shuttle_bus
    EIGHTEEN_SEATER_SHUTTLE_BUS = 4  # shuttle_bus
    BUS = 5  # bus
    TWELVE_SEATER_CAR = 6  # seater_12_16
    BIKE = 7  # bike
    TRUCK = 8  # truck

    labels = ['unknown', 'moto', 'car', '5 to 7 seater shuttle bus', '18 seater shuttle bus',
              'bus', '12 seater car', 'bike', 'truck']


class VehicleType(EnumType):
    UNKNOWN = 0
    MOTO = 1                # moto
    CAR = 2                 # car
    SHUTTER_BUS = 3         # shuttle_bus
    SHUTTER_BUS_18 = 4      # 18 seater shutter bus, NOT USED
    BUS = 5                 # bus
    SEATER_12_16 = 6        # seater_12_16
    BIKE = 7                # bike
    TRUCK = 8               # truck
    CART = 9                # cart
    MOTOR_RIDER = 10        # motor_rider
    CYCLIST = 11            # cyclist

    labels = ['unknown', 'moto', 'car', 'shuttle bus', '18 seater shuttle bus',
              'bus', '12 seater car', 'bike', 'truck', 'cart', 'motor_rider', 'cyclist']


class MovingDirection(EnumType):
    UNKNOWN = 0
    NORTH = 1
    NORTH_EAST = 2
    EAST = 3
    SOUTH_EAST = 4
    SOUTH = 5
    SOUTH_WEST = 6
    WEST = 7
    NORTH_WEST = 8

    labels = ['unknown', '0', '45', '90', '135', '180', '225', '270', '315']


class LicensePlateType(EnumType):
    UNKNOWN = 0
    COMMERCIAL = 1
    DIPLOMAT = 2
    FOREIGNER_OWNED = 3  # not used
    GOVERNMENT = 4
    MILITARY = 5
    PRIVATE = 6

    labels = ['unknown', 'commercial', 'diplomat', 'foreigner_owned', 'government', 'military', 'private']


class LicensePlateQuality(Quality):
    # license plate quality score will be split into 3 range good, medium, bad
    # NO_PLATE only exist when license_plate_text is None
    # place NO_PLATE into this enum for matching with UI filter in frontend
    UNKNOWN = 0
    GOOD = 1
    MEDIUM = 2
    BAD = 3
    NO_PLATE = 4

    labels = ['unknown', 'good', 'medium', 'bad', 'no_plate']


class RuleStatus(EnumType):
    UNKNOWN = 0
    ENABLED = 1
    DISABLED = 2
    DONE = 3


class RuleType(EnumType):
    UNKNOWN = 0
    ALERT = 1
    COUNTING = 2
    TRANSPORTATION = 3


class RuleCategory(EnumType):
    UNKNOWN = 0

    # for rule type: transportation
    OVERLOADING = 1
    TRAFFIC_LIGHT = 2
    NO_HELMET = 3
    WRONG_DIRECTION = 4
    WRONG_PARKING = 5
    RESTRICTED_WAY = 6
    WRONG_LANE = 7
    U_TURN_VIOLENCE = 8
    ONE_WAY_VIOLENCE = 9
    ROAD_LINE_VIOLENCE = 10
    OVER_SPEED_VIOLENCE = 11
    UNDER_SPEED_VIOLENCE = 12


class RulePriority(EnumType):
    UNKNOWN = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

    labels = ['unknown', 'high', 'medium', 'low']


class ScheduleType(EnumType):
    UNKNOWN = 0
    CONTINUOUS = 1
    DAILY = 2
    WEEKLY = 3


class ReceiverType(EnumType):
    UNKNOWN = 0
    USER = 1
    USER_GROUP = 2
    BROADCAST_CHANNEL = 3


class ChannelType(EnumType):
    UNKNOWN = 0
    TELEGRAM = 1
    ZALO = 2
    MESSENGER = 3

    labels = ['unknown', 'telegram', 'zalo', 'messenger']


class BroadcastChannelType(EnumType):
    UNKNOWN = 0
    TELEGRAM = 1
    ZALO = 2
    MESSENGER = 3

    labels = ['unknown', 'telegram', 'zalo', 'messenger']


class ChargingStationStatus(EnumType):
    UNKNOWN = 0
    FULL = 1
    FULL_AND_UNKNOWN = 2
    NOT_FULL = 3
    ABNORMAL_SLOTS = 4


class HelmetStatus(EnumType):
    UNKNOWN = 0
    NO_HELMET = 1
    HELMET = 2
    CYCLIST = 3
    HAT = 4


class VehicleOverloadStatus(EnumType):
    UNKNOWN = 0
    OVERLOADED = 1
    NO_OVERLOADED = 2


class AreaDirection(EnumType):
    UNKNOWN = 0
    IN = 1
    OUT = 2
    ANY = 3


class ExternalSystem(EnumType):
    UNKNOWN = 0
    MK_VISION = 1


class RuleProcessorType(EnumType):
    UNKNOWN = 0
    CENTRAL = 1
    EDGE = 2


class AlertResolutionStatusType(EnumType):
    UNKNOWN = 0
    WAITING = 1
    ACCEPTED = 2
    REJECTED = 3


class ModelProviderName:
    CHAT_OPEN_AI = 'ChatOpenAI'
    CHAT_OLLAMA = 'ChatOllama'


class PromptInputFormatType(EnumType):
    UNKNOWN = 0
    MANY_IMAGES = 1
    ONE_TEXT_MANY_IMAGES = 2

    labels = ['unknown', 'many_images', 'one_text_and_many_images']


class IVAStatus(EnumType):
    UNKNOWN = 0
    ENABLED = 1
    DISABLED = 2
