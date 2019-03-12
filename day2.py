import datetime


class CastleKilmereMember:
    """
    Creates a member of the Castle Kilmere School of Magic
    """

    def __init__(self, name, birthyear, sex,height=0):
        self._name = name
        self.birthyear = birthyear
        self.sex = sex
        self._height = height;#设置身高

    def says(self, words):
        return f"{self._name} says {words}"
    # property是只读属性
    # @property
    def age(self):
        now = datetime.datetime.now().year
        return now - self.birthyear

    def _get_height(self):
        return self._height

    def _set_height(self,height:int):#设置height的数据类型
         self._height = height

    # 以下只设置了getter
    # age = property(age)
    #如果调用print(bromley.age)方法，那么会报错，因为get为None
    age = property(None)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name}, birthyear: {self.birthyear})'

    @staticmethod
    def school_headmaster():
        return CastleKilmereMember('Redmond Dalodore', 1939, 'male')


class Pupil(CastleKilmereMember):
    """
    Create a Castle Kilmere Pupil
    """

    def __init__(self, name, birthyear, sex, house, start_year, pet=None):
        super().__init__(name, birthyear, sex)
        self.house = house
        self.start_year = start_year

        if pet is not None:
            self.pet_name, self.pet_type = pet


        self._elms = {
                  'Broomstick Flying': False,
                  'Art': False,
                  'Magical Theory': False,
                  'Foreign Magical Systems': False,
                  'Charms': False,
                  'Defence Against Dark Magic': False,
                  'Divination': False,
                  'Herbology': False,
                  'History of Magic': False,
                  'Potions': False,
                  'Transfiguration': False}

    @classmethod
    def cleon(cls):
        return cls('Cleon Bery', 2008, 'male', 'House of Courage', 2018, ('Cotton', 'owl'))

    @classmethod
    def flynn(cls):
        return cls('Flynn Gibbs', 2008, 'male', 'House of Courage', 2018, ('Twiggles', 'owl'))

    @classmethod
    def cassidy(cls):
        return cls('Cassidy', 2007, 'female', 'House of Courage', 2018, ('Ramses', 'cat'))

class Professor(CastleKilmereMember):
        """
        Creates a Castle Kilmere professor
        """

        def __init__(self, name:str, birthyear:int, sex:str, subject:str, house=None):
            super().__init__(name, birthyear, sex)
            self.subject = subject
            if house is not None:
                self.house = house

        @classmethod
        def mirren(cls):
            return cls('Miranda Mirren', 1963, 'female', 'Transfiguration', 'House of Courage')

        @classmethod
        def blade(cls):
            return cls('Blade Bardock', 1988, 'male', 'Potions', 'House of Ambition')

class Ghost(CastleKilmereMember):
        """
        Creates a Castle Kilmere ghost
        """

        def __init__(self, name, birthyear, sex, year_of_death, house=None):
            super().__init__(name, birthyear, sex)
            self.year_of_death = year_of_death

            if house is not None:
                self.house = house


if __name__ == "__main__":
        # bromley = CastleKilmereMember(name='Bromley Huckabee', birthyear=1959, sex='male')
        # cleon = Pupil(name='Cleon Bery', birthyear=2008, sex='male', house='House of Courage', start_year=2018)
        # headmaster = cleon.school_headmaster()
        #
        # mirren = Professor.mirren()
        # print(mirren.birthyear)
        # print(mirren.blade().house)
        # print(mirren.mirren().house)
        # print(mirren.school_headmaster().birthyear)
        # blade = Professor.blade()
        # cleon = Pupil.cleon()
        # flynn = Pupil.flynn()
        # cassidy = Pupil.cassidy()
        bromley = CastleKilmereMember(name='Bromley Huckabee', birthyear=1959, sex='male',height=123)
        bromley._set_height(122222)#修改height的属性
        # print(bromley.age)
        # bromley.age = 112#can't set attribute
        print(bromley._get_height())


