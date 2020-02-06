from typing import Tuple, Mapping, Any, Optional, List, Dict
import numpy as np
import tensorflow as tf
import nn
import random
import heapq
from datetime import datetime
import dateutil.parser # (When Python 3.7 becomes available, omit this line and use datetime.fromisoformat where needed)


PROFILE_SIZE = 24
DATA_PORTION = 16
OCCURRENCE = 2. # A constant float that falls outside the range [-1, 1] to indicate no rating value
DATA_OUTPUT_SIZE = 6

hash_seed = np.array([1319, 3779, 2027, 2459, 1523, 2777], dtype = np.uint16)
hash_scalar = np.array([47, 17, 31, 23, 19, 29], dtype = np.uint16)


class PairModel:
    def __init__(self) -> None:
        self.common_size = 48
        self.batch_size = 64
        self.batch_user = tf.Variable(np.zeros([self.batch_size, PROFILE_SIZE]), dtype = tf.float32)
        self.batch_item = tf.Variable(np.zeros([self.batch_size, PROFILE_SIZE]), dtype = tf.float32)
        self.digest_layer = nn.LayerLinear(PROFILE_SIZE, self.common_size)
        self.common_layer = nn.LayerLinear(self.common_size, 1)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-5)
        self.params = self.digest_layer.params + self.common_layer.params

    def set_users(self, user_profiles: tf.Tensor) -> None:
        self.batch_user.assign(user_profiles)

    def set_items(self, item_profiles: tf.Tensor) -> None:
        self.batch_item.assign(item_profiles)

    def act(self) -> tf.Tensor:
        user = self.digest_layer.act(self.batch_user)
        user = tf.nn.elu(user)
        item = self.digest_layer.act(self.batch_item)
        item = tf.nn.elu(item)
        common = user * item
        common = self.common_layer.act(common)
        return common

    def cost(self, targ: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.reduce_sum(tf.square(targ - pred), axis = 1), axis = 0)

    def refine(self, y: tf.Tensor) -> None:
        self.optimizer.minimize(lambda: self.cost(y, self.act()), self.params)

    def marshall(self) -> Mapping[str, Any]:
        return {
                "params": [ p.numpy().tolist() for p in self.params ],
            }

    def unmarshall(self, ob: Mapping[str, Any]) -> None:
        params = ob['params']
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of params')
        for i in range(len(params)):
            self.params[i].assign(np.array(params[i]))


class AttrModel:
    def __init__(self, pair_model: PairModel) -> None:
        self.pair_model = pair_model
        self.layer = nn.LayerLinear(DATA_PORTION, DATA_OUTPUT_SIZE)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-5)
        self.params = self.layer.params
        self.properties: List[Tuple[int, str]] = [] # profile_index, value
        self.instances = 0
        self.sum = np.zeros([DATA_OUTPUT_SIZE])
        self.sum_of_squares = np.zeros([DATA_OUTPUT_SIZE])

    def act(self) -> tf.Tensor:
        y = self.layer.act(self.pair_model.batch_user[:, :DATA_PORTION])
        y = tf.nn.tanh(y)
        return y

    def cost(self, targ: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.reduce_sum(tf.square(targ - pred), axis = 1), axis = 0)

    def refine(self, y: tf.Tensor) -> None:
        self.optimizer.minimize(lambda: self.cost(y, self.act()), self.params)

    def marshall(self) -> Mapping[str, Any]:
        return {
                "params": [ p.numpy().tolist() for p in self.params ],
            }

    def unmarshall(self, ob: Mapping[str, Any]) -> None:
        params = ob['params']
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of params')
        for i in range(len(params)):
            self.params[i].assign(np.array(params[i]))

    def str_to_vec(self, s: str) -> np.ndarray:
        try:
            # Try to interpret as a floating point value
            v = float(s)
            return np.full([6], v, dtype = np.float64)
        except ValueError:
            try:
                # Try as an iso date
                dt = dateutil.parser.parse(s)
                #dt = datetime.fromisoformat(s) # Use this when Python 3.7 comes out
                return np.array([dt.year, dt.month, dt.day, dt.weekday(), dt.hour, 60 * dt.minute + dt.second], dtype = np.float64)
            except ValueError:
                # Hash the string to a vector of six 16-bit values
                hash = hash_seed
                for i in range(len(s)):
                    hash *= hash_scalar
                    hash = np.bitwise_xor(hash, ord(s[i]), dtype = np.uint16)
                return hash.astype(np.float64)


class Profile:
    # role: -1=user, 0=neither, 1=item, 2=both
    def __init__(self, name: str, role: int) -> None:
        self.name = name
        self.values = np.random.normal(0., 0.01, [PROFILE_SIZE])
        self.role = role
        self.pairs: List[int] = []

    def set_role(self, role: int) -> None:
        if self.role == role or role == 0:
            return
        elif self.role == 0:
            self.role = role
        self.role = 2



class Engine:
    def __init__(self) -> None:
        self.pair_model = PairModel()
        self.profiles: List[Profile] = []
        self.attrs: List[AttrModel] = []
        self.pairs: List[Tuple[int, int, float]] = [] # user_index, item_index, rating
        self.str_to_profile_index: Dict[str, int] = {}
        self.str_to_attr_index: Dict[str, int] = {}

        # Buffers for batch training
        self.batch_users = np.empty([self.pair_model.batch_size, PROFILE_SIZE])
        self.batch_items = np.empty([self.pair_model.batch_size, PROFILE_SIZE])
        self.batch_ratings = np.empty([self.pair_model.batch_size, 1])
        self.batch_data = np.empty([self.pair_model.batch_size, DATA_OUTPUT_SIZE])

    # Input: name (or id string) for a profile
    # Output: The index and profile object associated with the id.
    #         (If no profile is associated with that id, one will be created.)
    def _get_profile(self, name: str, role: int) -> Tuple[int, Profile]:
        if name in self.str_to_profile_index:
            i = self.str_to_profile_index[name]
            p = self.profiles[i]
            p.set_role(role)
        else:
            i = len(self.profiles)
            p = Profile(name, role)
            self.profiles.append(p)
            self.str_to_profile_index[name] = i
        return i, p

    # Input: name (or id string) of an attribute
    # Output: The attribute model for that id.
    #         (If no attribute model is associated with that id, one will be created.)
    def _get_attr_model(self, name: str) -> Tuple[int, AttrModel]:
        if name in self.str_to_attr_index:
            i = self.str_to_attr_index[name]
            a = self.attrs[i]
        else:
            i = len(self.attrs)
            a = AttrModel(self.pair_model)
            self.attrs.append(a)
            self.str_to_attr_index[name] = i
        return i, a

    # Performs one batch of training on the pair model
    def _train_pairs(self, num_preferred: int = 0, preferred: List[int] = []) -> None:
        # Make a batch
        pair_indexes = []
        for i in range(self.batch_users.shape[0]):
            if i < num_preferred:
                index = preferred[random.randrange(len(preferred))]
            else:
                index = random.randrange(len(self.pairs))
            pair_indexes.append(index)
            pair = self.pairs[index]
            user_index = pair[0]
            item_index = pair[1]
            self.batch_users[i] = self.profiles[user_index].values
            self.batch_items[i] = self.profiles[item_index].values
            if pair[2] == OCCURRENCE:
                r = random.randrange(4)
                if r < 2:
                    # Make a negative occurrence example
                    if r == 0:
                        self.batch_users[i] = self.profiles[random.randrange(len(self.profiles))].values
                    else:
                        self.batch_items[i] = self.profiles[random.randrange(len(self.profiles))].values
                    self.batch_ratings[i, 0] = 0.
                else:
                    self.batch_ratings[i, 0] = 1.
            else:
                self.batch_ratings[i, 0] = pair[2]

        # Refine
        self.pair_model.set_users(self.batch_users)
        self.pair_model.set_items(self.batch_items)
        self.pair_model.refine(self.batch_ratings)

        # Store changes
        updated_users = self.pair_model.batch_user.numpy()
        updated_items = self.pair_model.batch_item.numpy()
        for i in range(len(pair_indexes)):
            index = pair_indexes[i]
            pair = self.pairs[index]
            user_index = pair[0]
            item_index = pair[1]
            self.profiles[user_index].values = updated_users[i]
            self.profiles[item_index].values = updated_items[i]

    # Performs one batch of training on an AttrModel
    def _train_attr(self, attr: AttrModel) -> None:
        # Make a batch
        property_indexes = []
        mean = attr.sum / attr.instances
        inv_dev = 1. / np.sqrt(np.maximum((attr.sum_of_squares / attr.instances) - (mean * mean), 1e-16))
        for i in range(self.batch_users.shape[0]):
            index = random.randrange(len(attr.properties))
            property_indexes.append(index)
            property = attr.properties[index]
            self.batch_users[i] = self.profiles[property[0]].values
            self.batch_data[i] = (attr.str_to_vec(property[1]) - mean) * inv_dev

        # Refine
        attr.pair_model.set_users(self.batch_users)
        attr.refine(self.batch_data)

        # Store changes
        updated_profiles = self.pair_model.batch_user.numpy()
        for i in range(len(property_indexes)):
            index = property_indexes[i]
            property = attr.properties[index]
            self.profiles[property[0]].values = updated_profiles[i]

    # Call this when a user interacts with an item
    def addOccurrence(self, user: str, item: str) -> None:
        i_user, p_user = self._get_profile(user, -1)
        i_item, p_item = self._get_profile(item, 1)
        i_pair = len(self.pairs)
        self.pairs.append((i_user, i_item, OCCURRENCE))
        p_user.pairs.append(i_pair)
        p_item.pairs.append(i_pair)
        if len(self.pairs) > 64:
            self._train_pairs()

    # Call this when a user expresses a rating for an item
    def addOpinion(self, user: str, item: str, opinion: float) -> None:
        i_user, p_user = self._get_profile(user, -1)
        i_item, p_item = self._get_profile(item, 1)
        i_pair = len(self.pairs)
        self.pairs.append((i_user, i_item, opinion))
        p_user.pairs.append(i_pair)
        p_item.pairs.append(i_pair)
        if len(self.pairs) > 64:
            self._train_pairs()

    # Call this to describe a particular user or item by providing an attribute name and value
    def addProperty(self, id: str, attr_name: str, value: str) -> None:
        i_profile, profile = self._get_profile(id, 0)
        i_attr, attr = self._get_attr_model(attr_name)
        i_property = len(attr.properties)
        attr.properties.append((i_profile, value))
        if attr.instances > 1000:
            # Drop an average instance so recent values will have more impact
            mean = attr.sum / attr.instances
            attr.sum -= mean
            attr.sum_of_squares -= mean * mean
            attr.instances -= 1
        vals = attr.str_to_vec(value)
        attr.sum += vals
        attr.sum_of_squares += vals * vals
        attr.instances += 1
        if len(attr.properties) > 64:
            self._train_attr(attr)

    def recommend(self, id: str, num_recommendations: int) -> List[Tuple[str, float]]:
        # Do some training
        i_profile, profile = self._get_profile(id, 0)
        for i in range(16):
            if len(self.attrs) > 0:
                attr = self.attrs[random.randrange(len(self.attrs))]
                if len(attr.properties) > 8:
                    self._train_attr(attr)
            if len(profile.pairs) > 4:
                self._train_pairs(20, profile.pairs)
            else:
                self._train_pairs()

        # Predict the fitness for a bunch of random profiles
        self.batch_users[:] = profile.values
        pri_q: List[Tuple[float, int]] = []
        for j in range(16):
            item_indexes = []
            patience = 6 * self.batch_users.shape[0]
            for i in range(self.batch_users.shape[0]):
                while patience > 0:
                    prof_index = random.randrange(len(self.profiles))
                    prof = self.profiles[prof_index]
                    if prof.role != profile.role or prof.role == 2:
                        break
                    patience -= 1
                item_indexes.append(prof_index)
                self.batch_items[i] = prof.values
            if patience > 0:
                self.pair_model.set_users(self.batch_users)
                self.pair_model.set_items(self.batch_items)
                pred = self.pair_model.act().numpy()
                for i in range(self.batch_users.shape[0]):
                    heapq.heappush(pri_q, (-pred[i, 0], item_indexes[i]))

        # Return the best ones
        recommendations: List[Tuple[str, float]] = []
        for i in range(num_recommendations):
            tup = heapq.heappop(pri_q)
            recommendations.append((self.profiles[tup[1]].name, -tup[0]))
        return recommendations
