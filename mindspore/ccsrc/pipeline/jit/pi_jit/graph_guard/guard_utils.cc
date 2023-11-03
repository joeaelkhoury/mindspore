/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pipeline/jit/pi_jit/graph_guard/guard_utils.h"
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "pybind_api/ir/cell_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi_jit/utils/utils.h"

namespace mindspore {
namespace jit {
namespace graph {
static std::string GetObjectString(PyObject *objName) {
  std::string ret = "";
  if (objName == NULL) {
    return ret;
  }
  PyObject *pyName = PyUnicode_AsEncodedString(objName, "utf-8", NULL);
  char *strName = PyBytes_AsString(pyName);
  if (strName != nullptr) {
    ret = strName;
  }
  Py_DECREF(pyName);
  return ret;
}

#define DESC(op) (std::string("{") + std::string(#op) + std::string(":") + (op) + std::string("}"))
#define DESC_STRING(op) (std::string("{") + std::string(#op) + std::string(":") + std::to_string(op) + std::string("}"))
#define DESC_STRING_L(op, l)                                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(l) + std::string("]") + std::string(":") + \
   std::to_string(op) + std::string("}"))  // NOLINT
#define DESC_STRING_S(op, l)                                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(l) + std::string("]") + std::string(":") + \
   (op) + std::string("}"))  // NOLINT
#define DESC_STRING_O(obj, op) \
  (std::string("{") + std::string(#op) + std::string(":") + std::to_string(obj->op) + std::string("}"))
#define DESC_TOSTRING(op)                                                                                             \
  (std::string("{") + std::string(#op) + std::string(":") + ((op == nullptr) ? std::string("nil") : op->ToString()) + \
   std::string("}"))  // NOLINT
#define DESC_ITEM(opK, opV)                                                                          \
  (std::string("{") + ((opK == nullptr) ? std::string("nil") : opK->ToString()) + std::string(":") + \
   ((opV == nullptr) ? std::string("nil") : opV->ToString()) + std::string("}"))  // NOLINT
#define DESC_ITEM_V(op) (std::string("{") + std::to_string(op) + std::string("}"))
#define DESC_ITEM_T(op) (std::string("{") + ((op == nullptr) ? std::string("nil") : op->ToString()) + std::string("}"))
#define DESC_INDEX(op, idx)                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(idx) + std::string("]") + \
   std::string(":") + ((op[idx] == nullptr) ? std::string("nil") : op[idx]->ToString()) + std::string("}"))  // NOLINT
#define DESC_INDEX_V(op, idx)                                                                        \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(idx) + std::string("]") + \
   std::string(":") + std::to_string(op[idx]) + std::string("}"))  // NOLINT
#define DESC_END ItemData::ToString()

typedef enum _ItemType {
  PyNull = 0,
  PyLong,
  PyFloat,
  PyBool,
  PyBytes,
  PyStr,
  PyList,
  PyTuple,
  PySet,
  PyFrozenSet,
  PyDict,
  PyComplex,
  PySlice,
  PyFunction,
  PyMethod,
  PyInstanceMethod,
  PyType,
  PyNumpy,
  PyUnknown,
  TensorType,
  ParamInfo,
  MetaTensor,
  Tensor,
  MapTensor,
  RowTensor,
  COOTensor,
  CSRTensor,
  Tensordata,
  Primitive,
  Cell,
} ItemType;

class ItemData {
 public:
  ItemData(ItemType itemType, bool needSpecialize, int recurseDepth)
      : tp_(itemType), specialized_(needSpecialize), recurseDepth_(recurseDepth) {}

  virtual ~ItemData() = default;

  virtual bool operator==(const ItemData &obj) { return obj.tp_ == tp_; }

  virtual std::string ToString() {
    if (tp_ == ItemType::PyNull) {
      return "(null)";
    } else {
      return std::string("(type:") + std::to_string(static_cast<int>(tp_)) +
             ",specialize:" + std::to_string(specialized_) + ",recurse:" + std::to_string(recurseDepth_) + ")";
    }
  }

 protected:
  ItemType tp_;
  bool specialized_;
  int recurseDepth_;
};
using ItemDataPtr = std::shared_ptr<ItemData>;

static ItemDataPtr CreateItem(PyObject *obj, bool needSpecialize = true, int recurseDepth = INT_MAX);

class IntData : public ItemData {
 public:
  IntData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyLong, needSpecialize, recurseDepth) {
    tp_ = ItemType::PyLong;
    intVar_ = PyLong_AsLong(obj);
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) && (!specialized_ || (((const IntData &)obj).intVar_ == intVar_));
  }

  std::string ToString() override { return DESC_STRING(intVar_) + DESC_END; }

 protected:
  int64_t intVar_;
};

class FloatData : public ItemData {
 public:
  FloatData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyFloat, needSpecialize, recurseDepth) {
    floatVar_ = PyFloat_AsDouble(obj);
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) && (!specialized_ || ((const FloatData &)obj).floatVar_ == floatVar_);
  }

  std::string ToString() override { return DESC_STRING(floatVar_) + DESC_END; }

 protected:
  double floatVar_;
};

class BoolData : public ItemData {
 public:
  BoolData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyBool, needSpecialize, recurseDepth) {
    boolVar_ = (obj == Py_True);
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) && (!specialized_ || ((const BoolData &)obj).boolVar_ == boolVar_);
  }

  std::string ToString() override { return DESC_STRING(boolVar_) + DESC_END; }

 protected:
  bool boolVar_;
};

class BytesData : public ItemData {
 public:
  BytesData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyBytes, needSpecialize, recurseDepth), len_(PyBytes_Size(obj)) {
    if (needSpecialize) {
      buf_ = reinterpret_cast<char *>(malloc(len_ * sizeof(char)));
      if (buf_ != nullptr) {
        char *pBuf = PyBytes_AS_STRING(obj);
        if (pBuf != nullptr) {
          memcpy(buf_, pBuf, len_);
        } else {
          free(buf_);
          buf_ = nullptr;
        }
      }
    } else {
      buf_ = nullptr;
    }
  }

  virtual ~BytesData() {
    if (buf_ != nullptr) {
      free(buf_);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const BytesData &other = (const BytesData &)obj;
      return len_ == other.len_ && ((specialized_ && (len_ == 0 || (buf_ != nullptr && other.buf_ != nullptr &&
                                                                    memcmp(buf_, other.buf_, len_) == 0))) ||
                                    (!specialized_));
    }
    return false;
  }

  std::string ToString() override {
    size_t bytes = (size_t)buf_;
    return DESC_STRING_L(bytes, len_) + DESC_END;
  }

 protected:
  Py_ssize_t len_;
  char *buf_ = nullptr;
};

class StringData : public ItemData {
 public:
  StringData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyStr, needSpecialize, recurseDepth) {
    if (needSpecialize) {
      strVal_ = GetObjectString(obj);
    }
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) &&
           ((specialized_ && ((const StringData &)obj).strVal_.compare(strVal_) == 0) || (!specialized_));
  }

  std::string ToString() override { return DESC(strVal_) + DESC_END; }

 protected:
  std::string strVal_;
};

class ListData : public ItemData {
 public:
  ListData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyList, needSpecialize, recurseDepth) {
    if (PyList_Check(obj)) {
      tp_ = ItemType::PyList;
      for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if (item != NULL) {
          if (recurseDepth > 0 || needSpecialize) {
            listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
          } else {
            listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
          }
        }
      }
    } else if (PyTuple_Check(obj)) {
      tp_ = ItemType::PyTuple;
      for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(obj); ++i) {
        PyObject *item = PyTuple_GET_ITEM(obj, i);
        if (item != NULL) {
          if (recurseDepth > 0 || needSpecialize) {
            listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
          } else {
            listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
          }
        }
      }
    } else if (PySet_Check(obj)) {
      tp_ = ItemType::PySet;
      Py_ssize_t pos = 0;
      PyObject *item;
      Py_hash_t hash;
      while (_PySet_NextEntry(obj, &pos, &item, &hash)) {
        if (recurseDepth > 0 || needSpecialize) {
          listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
        } else {
          listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
        }
      }
      inOrder_ = false;
    } else if (PyFrozenSet_Check(obj)) {
      tp_ = ItemType::PyFrozenSet;
      Py_ssize_t pos = 0;
      PyObject *item;
      Py_hash_t hash;
      while (_PySet_NextEntry(obj, &pos, &item, &hash)) {
        if (recurseDepth > 0 || needSpecialize) {
          listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
        } else {
          listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
        }
      }
      inOrder_ = false;
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const ListData &list = (const ListData &)obj;
      if (list.listVar_.size() == listVar_.size()) {
        if (!inOrder_) {
          std::vector<ItemDataPtr> listCpy = list.listVar_;
          for (size_t i = 0, j; i < listVar_.size(); ++i) {
            size_t lenList = listCpy.size();
            for (j = 0; j < lenList; ++j) {
              if (*(listCpy[j]) == *(listVar_[i])) {
                listCpy.erase(listCpy.begin() + j);
                break;
              }
            }
            if (j == lenList) {
              return false;
            }
          }
        } else {
          for (size_t i = 0; i < listVar_.size(); ++i) {
            if (*(list.listVar_[i]) == *(listVar_[i])) {
              continue;
            } else {
              return false;
            }
          }
        }
        return true;
      }
    }
    return false;
  }

  std::string ToString() override {
    std::string ret;
    for (auto it : listVar_) {
      ret += DESC_ITEM_T(it);
    }
    switch (tp_) {
      case ItemType::PyList: {
        std::string list = ret;
        ret = DESC_STRING_S(list, listVar_.size());
      } break;
      case ItemType::PyTuple: {
        std::string tuple = ret;
        ret = DESC_STRING_S(tuple, listVar_.size());
      } break;
      case ItemType::PySet: {
        std::string set = ret;
        ret = DESC_STRING_S(set, listVar_.size());
      } break;
      case ItemType::PyFrozenSet: {
        std::string fronzen_set = ret;
        ret = DESC_STRING_S(fronzen_set, listVar_.size());
      } break;
      default:
        ret = "unknown";
        break;
    }
    return ret + DESC_END;
  }

 protected:
  std::vector<ItemDataPtr> listVar_;
  bool inOrder_ = true;
};

class ComplexData : public ItemData {
 public:
  ComplexData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyComplex, needSpecialize, recurseDepth) {
    if (needSpecialize) {
      complexVar_ = std::make_pair(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
    }
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) && (!specialized_ || ((const ComplexData &)obj).complexVar_ == complexVar_);
  }

  std::string ToString() override {
    return "complex(" + std::to_string(complexVar_.first) + "," + std::to_string(complexVar_.second) + ")" + DESC_END;
  }

 protected:
  std::pair<double, double> complexVar_;
};

class SliceData : public ItemData {
 public:
  SliceData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PySlice, needSpecialize, recurseDepth) {
    Py_ssize_t start = 0, stop = 0, step = 0;
    if (needSpecialize) {
      PySlice_Unpack(obj, &start, &stop, &step);
      sliceVar_.push_back(start);
      sliceVar_.push_back(stop);
      sliceVar_.push_back(step);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const SliceData &other = (const SliceData &)obj;
      return (!specialized_ || (other.sliceVar_[0] == sliceVar_[0] && other.sliceVar_[1] == sliceVar_[1] &&
                                other.sliceVar_[2] == sliceVar_[2]));
    }
    return false;
  }

  std::string ToString() override {
    std::string slice;
    for (auto it : sliceVar_) {
      slice += DESC_ITEM_V(it);
    }
    return DESC_STRING_S(slice, sliceVar_.size()) + DESC_END;
  }

 protected:
  std::vector<Py_ssize_t> sliceVar_;
};

class DictData : public ItemData {
 public:
  DictData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyDict, needSpecialize, recurseDepth) {
    Py_ssize_t pos = 0;
    PyObject *key, *val;
    while (PyDict_Next(obj, &pos, &key, &val)) {
      ItemDataPtr k, v;
      if (recurseDepth > 0 || needSpecialize) {
        k = CreateItem(key, needSpecialize, recurseDepth);
        v = CreateItem(val, needSpecialize, recurseDepth);
      } else {
        k = CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(key)), false, false);
        v = CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(val)), false, false);
      }
      listK_.push_back(k);
      listV_.push_back(v);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const DictData &other = (const DictData &)obj;
      if (other.listK_.size() == listK_.size() && other.listV_.size() == listV_.size()) {
        std::vector<ItemDataPtr> listCpK = other.listK_;
        std::vector<ItemDataPtr> listCpV = other.listV_;
        for (size_t i = 0, j = 0; i < listK_.size(); ++i) {
          size_t lenList = listCpK.size();
          for (; j < lenList; ++j) {
            if (*(listK_[i]) == *(listCpK[j]) && *(listV_[i]) == *(listCpV[j])) {
              listCpK.erase(listCpK.begin() + j);
              listCpV.erase(listCpV.begin() + j);
              break;
            }
          }
          if (j == lenList) {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  std::string ToString() override {
    std::string dict;
    for (size_t i = 0; i < listK_.size(); ++i) {
      dict += DESC_ITEM(listK_[i], listV_[i]);
    }
    return DESC_STRING_S(dict, listK_.size()) + DESC_END;
  }

 protected:
  std::vector<ItemDataPtr> listK_;
  std::vector<ItemDataPtr> listV_;
};

class FunctionData : public ItemData {
 public:
  FunctionData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyFunction, needSpecialize, recurseDepth) {
    if (needSpecialize || recurseDepth > 0) {
      code_ = reinterpret_cast<PyCodeObject *>(PyFunction_GetCode(obj));
      globals_ = CreateItem(PyFunction_GetGlobals(obj), needSpecialize, recurseDepth);
      module_ = CreateItem(PyFunction_GetModule(obj), needSpecialize, recurseDepth);
      defaults_ = CreateItem(PyFunction_GetDefaults(obj), needSpecialize, recurseDepth);
      kwdefaults_ = CreateItem(PyFunction_GetKwDefaults(obj), needSpecialize, recurseDepth);
      closure_ = CreateItem(PyFunction_GetClosure(obj), needSpecialize, recurseDepth);
    } else {
      code_ = reinterpret_cast<PyCodeObject *>(PyFunction_GetCode(obj));
      PyObject *temp = PyFunction_GetGlobals(obj);
      globals_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetModule(obj);
      module_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetDefaults(obj);
      defaults_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetKwDefaults(obj);
      kwdefaults_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetClosure(obj);
      closure_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const FunctionData &other = (const FunctionData &)obj;
      return code_ == other.code_ && *globals_ == *(other.globals_) && *module_ == *(other.module_) &&
             *defaults_ == *(other.defaults_) && *kwdefaults_ == *(other.kwdefaults_) && *closure_ == *(other.closure_);
    }
    return false;
  }

  std::string ToString() override {
    std::string func = DESC_TOSTRING(globals_) + DESC_TOSTRING(module_) + DESC_TOSTRING(defaults_) +
                       DESC_TOSTRING(kwdefaults_) + DESC_TOSTRING(closure_);
    return DESC(func) + DESC_END;
  }

 protected:
  PyCodeObject *code_;
  ItemDataPtr globals_;
  ItemDataPtr module_;
  ItemDataPtr defaults_;
  ItemDataPtr kwdefaults_;
  ItemDataPtr closure_;
};

class MethodData : public ItemData {
 public:
  MethodData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyMethod, needSpecialize, recurseDepth),
        refFunc_(CreateItem(PyMethod_GET_FUNCTION(obj), needSpecialize, recurseDepth)),
        refSelf_(CreateItem(PyMethod_GET_SELF(obj), needSpecialize, recurseDepth)) {}

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const MethodData &other = (const MethodData &)obj;
      return *refFunc_ == *(other.refFunc_) && *refSelf_ == *(other.refSelf_);
    }
    return false;
  }

  std::string ToString() override {
    std::string method = DESC_TOSTRING(refFunc_) + DESC_TOSTRING(refSelf_);
    return DESC(method) + DESC_END;
  }

 protected:
  ItemDataPtr refFunc_;
  ItemDataPtr refSelf_;
};

class InstanceMethodData : public ItemData {
 public:
  InstanceMethodData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyInstanceMethod, needSpecialize, recurseDepth),
        refFunc_(CreateItem(PyInstanceMethod_GET_FUNCTION(obj), needSpecialize, recurseDepth)) {}

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const InstanceMethodData &other = (const InstanceMethodData &)obj;
      return *refFunc_ == *(other.refFunc_);
    }
    return false;
  }

  std::string ToString() override {
    std::string instance_method = DESC_TOSTRING(refFunc_);
    return DESC(instance_method) + DESC_END;
  }

 protected:
  ItemDataPtr refFunc_;
};

class TypeData : public ItemData {
 public:
  TypeData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyType, needSpecialize, recurseDepth) {
    refType_ = reinterpret_cast<PyTypeObject *>(obj);
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      return refType_ == ((const TypeData &)obj).refType_;
    }
    return false;
  }

  std::string ToString() override {
    std::string type = refType_->tp_name;
    return DESC(type) + DESC_END;
  }

 protected:
  PyTypeObject *refType_;
};

class NumpyData : public ItemData {
 public:
  NumpyData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyNumpy, needSpecialize, recurseDepth) {
    py::array arr = py::cast<py::array>(obj);
    dtype_ = arr.dtype();
    size_ = arr.size();
    itemsize_ = arr.itemsize();
    ndim_ = arr.ndim();
    nbytes_ = arr.nbytes();
    for (ssize_t i = 0; i < ndim_; ++i) {
      shape_.push_back(arr.shape()[i]);
      strides_.push_back(arr.strides()[i]);
    }
    if (arr.data() != nullptr) {
      if (needSpecialize) {
        buf_ = reinterpret_cast<char *>(malloc(sizeof(char) * nbytes_));
        if (buf_ != NULL) {
          memcpy(buf_, arr.data(), nbytes_);
        }
      } else {
        buf_ = NULL;
      }
    } else {
      buf_ = NULL;
    }
  }

  virtual ~NumpyData() {
    if (buf_ != NULL && specialized_) {
      free(buf_);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const NumpyData &other = (const NumpyData &)obj;
      return dtype_ == other.dtype_ && size_ == other.size_ && ndim_ == other.ndim_ && nbytes_ == other.nbytes_ &&
             shape_ == other.shape_ && strides_ == other.strides_ &&
             (!specialized_ || (buf_ != NULL && other.buf_ != NULL && memcmp(buf_, other.buf_, nbytes_) == 0));
    }
    return false;
  }

  std::string ToString() override {
    std::string numpy;
    char dtype_kind = dtype_.kind();
    numpy +=
      DESC_STRING(dtype_kind) + DESC_STRING(size_) + DESC_STRING(itemsize_) + DESC_STRING(ndim_) + DESC_STRING(nbytes_);
    for (size_t i = 0; i < shape_.size(); ++i) {
      numpy += DESC_INDEX_V(shape_, i) + DESC_INDEX_V(strides_, i);
    }
    return DESC(numpy) + DESC_END;
  }

 protected:
  py::dtype dtype_;
  ssize_t size_;
  ssize_t itemsize_;
  ssize_t ndim_;
  ssize_t nbytes_;
  std::vector<ssize_t> shape_;
  std::vector<ssize_t> strides_;
  char *buf_;
};

class TensorTypeData : public ItemData {
 public:
  TensorTypeData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::TensorType, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto ptr = pyObj.cast<mindspore::TypePtr>();
    tpp_ = ptr->Clone();
  }

  bool operator==(const ItemData &obj) override {
    return ItemData::operator==(obj) && (!specialized_ || *(((const TensorTypeData &)obj).tpp_) == *tpp_);
  }

  std::string ToString() override {
    std::string tensor_type = tpp_->ToString();
    return DESC(tensor_type) + DESC_END;
  }

 protected:
  mindspore::TypePtr tpp_;
};

class ParamInfoData : public ItemData {
 public:
  ParamInfoData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::ParamInfo, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto ptr = pyObj.cast<mindspore::ParamInfoPtr>();
    param_ = ptr->Clone();
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      if (!specialized_) {
        return true;
      }
      const ParamInfoData &other = (const ParamInfoData &)obj;
      return Equal(param_, other.param_);
    }
    return false;
  }

  static bool Equal(ParamInfoPtr a, ParamInfoPtr b) {
    return a->requires_grad() == b->requires_grad() && a->comm_fusion() == b->comm_fusion() &&
           a->parallel_optimizer() == b->parallel_optimizer() &&
           a->parallel_optimizer_comm_recompute() == b->parallel_optimizer_comm_recompute() &&
           a->parameter_shape() == b->parameter_shape() && a->use_persistent_storage() == b->use_persistent_storage() &&
           a->cache_enable() == b->cache_enable() && a->param_strategy() == b->param_strategy() &&
           a->cache_shape() == b->cache_shape() && a->requires_aggr() == b->requires_aggr();
  }

  std::string ToString() override {
    std::string param_info = ToStringAttr(param_);
    return DESC(param_info) + DESC_END;
  }

  static std::string ToStringAttr(mindspore::ParamInfoPtr p) {
    if (p == nullptr) {
      return "nil";
    }
    std::string param_name = p->name();
    std::string ret = DESC(param_name) + DESC_STRING_O(p, requires_grad()) + DESC_STRING_O(p, comm_fusion()) +
                      DESC_STRING_O(p, parallel_optimizer()) + DESC_STRING_O(p, requires_aggr()) +
                      DESC_STRING_O(p, parallel_optimizer_comm_recompute()) +
                      DESC_STRING_O(p, use_persistent_storage()) + DESC_STRING_O(p, cache_enable());
    auto parameter_shape = p->parameter_shape();
    for (size_t i = 0; i < parameter_shape.size(); ++i) {
      ret += DESC_INDEX_V(parameter_shape, i);
    }
    auto cache_shape = p->cache_shape();
    for (size_t i = 0; i < cache_shape.size(); ++i) {
      ret += DESC_INDEX_V(cache_shape, i);
    }
    auto param_strategy = p->param_strategy();
    for (size_t i = 0; i < param_strategy.size(); ++i) {
      ret += DESC_INDEX_V(param_strategy, i);
    }
    return ret;
  }

 protected:
  mindspore::ParamInfoPtr param_;
};

class MetaTensorData : public ItemData {
 public:
  MetaTensorData(mindspore::tensor::MetaTensorPtr tensor_ptr, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::MetaTensor, needSpecialize, recurseDepth) {
    StoreTensor(tensor_ptr);
  }

  MetaTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::MetaTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    constexpr char const_arg_attr[] = "const_arg";
    if (!py::hasattr(pyObj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      specialized_ = needSpecialize;
    } else {
      specialized_ = true;
    }
    mindspore::tensor::MetaTensorPtr tensor_ptr;
    if (py::isinstance<mindspore::tensor::MapTensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    } else if (py::isinstance<mindspore::tensor::Tensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
    } else {
      tensor_ptr = pyObj.cast<mindspore::tensor::MetaTensorPtr>();
    }
    StoreTensor(tensor_ptr);
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const MetaTensorData &other = (const MetaTensorData &)obj;
      bool ret = tid_ == other.tid_ && shape_ == other.shape_ && format_.compare(other.format_) == 0 &&
                 host_format_.compare(other.host_format_) == 0 && is_parameter_ == other.is_parameter_ &&
                 ((data_type_ == nullptr && other.data_type_ == nullptr) ||
                  (data_type_ != nullptr && other.data_type_ != nullptr && *data_type_ == *(other.data_type_)));
      if (is_parameter_ == true) {
        ret = ret && ((param_ == nullptr && other.param_ == nullptr) ||
                      (param_ != nullptr && other.param_ != nullptr && ParamInfoData::Equal(param_, other.param_)));
      }
      return ret;
    }
    return false;
  }

  std::string ToString() override {
    std::string meta_tensor = ToStringIntern();
    return DESC(meta_tensor) + DESC_END;
  }

 protected:
  virtual std::string ToStringIntern() {
    std::string param_desc = ParamInfoData::ToStringAttr(param_);
    std::string shape = "";
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape += DESC_INDEX_V(shape_, i);
    }
    return DESC_STRING(tid_) + DESC(format_) + DESC(host_format_) + DESC_TOSTRING(data_type_) +
           DESC_STRING(is_parameter_) + DESC(param_desc) + DESC(shape);
  }

  void StoreTensor(mindspore::tensor::MetaTensorPtr tensor_ptr) {
    tid_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    mindspore::tensor::DeviceInfo info = tensor_ptr->device_info();
    format_ = info.format_;
    host_format_ = info.host_format_;
    data_type_ = info.data_type_ != nullptr ? info.data_type_->Clone() : nullptr;
    is_parameter_ = tensor_ptr->is_parameter();
    param_ = tensor_ptr->param_info() != nullptr ? tensor_ptr->param_info()->Clone() : nullptr;
  }

  mindspore::TypeId tid_;
  ShapeVector shape_;
  std::string format_;
  std::string host_format_;
  TypePtr data_type_;
  bool is_parameter_;
  mindspore::ParamInfoPtr param_;
};

class TensorData : public MetaTensorData {
 public:
  TensorData(mindspore::tensor::TensorPtr tensor_ptr, bool needSpecialize, int recurseDepth)
      : MetaTensorData(tensor_ptr, needSpecialize, recurseDepth) {
    tp_ = ItemType::Tensor;
    StoreTensor(tensor_ptr);
  }

  TensorData(PyObject *obj, bool needSpecialize, int recurseDepth) : MetaTensorData(obj, needSpecialize, recurseDepth) {
    tp_ = ItemType::Tensor;
    auto pyObj = py::cast<py::object>(obj);
    mindspore::tensor::TensorPtr tensor_ptr;
    if (py::isinstance<mindspore::tensor::MapTensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    } else {
      tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
    }
    StoreTensor(tensor_ptr);
  }

  bool IsBaseShapePtr(const TensorData &other) {
    return (other.base_shape_ptr_ == nullptr && base_shape_ptr_ == other.base_shape_ptr_) ||
           (base_shape_ptr_ != nullptr && other.base_shape_ptr_ != nullptr &&
            *(other.base_shape_ptr_) == *(base_shape_ptr_));
  }

  bool IsCastDtype(const TensorData &other) {
    return (other.cast_dtype_ == nullptr && cast_dtype_ == nullptr) ||
           (other.cast_dtype_ != nullptr && cast_dtype_ != nullptr && *cast_dtype_ == *(other.cast_dtype_));
  }

  bool operator==(const ItemData &obj) override {
    if (!ItemData::operator==(obj)) {
      return false;
    }
    bool ret = MetaTensorData::operator==(obj);
    const TensorData &other = (const TensorData &)obj;
    ret = ret && other.init_flag_ == init_flag_ && other.is_forward_output_ == is_forward_output_ &&
          /*other.id_.compare(id_) == 0 &&*/ other.graph_output_ == graph_output_ &&
          other.specialized_ == specialized_ && IsBaseShapePtr(other) && IsCastDtype(other) &&
          other.compression_type_ == compression_type_ && other.quant_params_.size() == quant_params_.size() &&
          other.tensor_name_.compare(tensor_name_) == 0;
    if (!ret) {
      return ret;
    }
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      if (*(quant_params_[i]) == *(other.quant_params_[i])) {
        continue;
      } else {
        return false;
      }
    }
    if (specialized_) {
      if (data_ == nullptr) {
        ret = data_ == other.data_;
      } else {
        ret = data_->equals(*(other.data_));
      }
    } else {
      ret = data_ == other.data_;
    }
    return ret;
  }

  std::string ToString() override {
    std::string tensor = ToStringIntern();
    return DESC(tensor) + DESC_END;
  }

 protected:
  std::string ToStringIntern() override {
    std::string ret = MetaTensorData::ToStringIntern();
    ret += DESC_STRING(is_forward_output_) + DESC_STRING(init_flag_) + DESC_STRING(graph_output_);
    ret +=
      DESC_TOSTRING(cast_dtype_) + DESC_TOSTRING(base_shape_ptr_) + DESC_STRING(compression_type_) + DESC(tensor_name_);
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      ret += DESC_INDEX(quant_params_, i);
    }
    return ret;
  }

  void StoreTensor(mindspore::tensor::TensorPtr tensor_ptr) {
    init_flag_ = tensor_ptr->is_init();
    is_forward_output_ = tensor_ptr->is_forward_output();
    id_ = tensor_ptr->id();
    graph_output_ = tensor_ptr->IsGraphOutput();
    base_shape_ptr_ = tensor_ptr->base_shape_ptr() == nullptr ? nullptr : tensor_ptr->base_shape_ptr()->Clone();
    cast_dtype_ = (tensor_ptr->cast_dtype() == nullptr) ? nullptr : tensor_ptr->cast_dtype()->Clone();
    compression_type_ = tensor_ptr->compression_type();
    const std::vector<std::shared_ptr<mindspore::QuantizationParam>> &qp = tensor_ptr->quant_params();
    tensor_name_ = tensor_ptr->name();
    for (auto quant : qp) {
      QuantizationParamPtr qptr = std::make_shared<mindspore::QuantizationParam>(quant->quant_algo_name());
      quant_params_.push_back(qptr);
      qptr->set_attrs(quant->attrs());
    }
    if (specialized_) {
      tensor_ptr->data_sync(true);
      data_ = tensor_ptr->data_ptr();
    }
  }

  bool init_flag_;
  bool is_forward_output_;
  mindspore::tensor::TensorDataPtr data_;
  std::string id_;
  bool graph_output_;
  // bool updated_by_device_{false};
  // DeviceSyncPtr device_sync_{nullptr};
  // bool need_release_device_mem_{false};
  // bool cache_enable_{false};
  mindspore::abstract::BaseShapePtr base_shape_ptr_;
  // std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  // std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  mindspore::TypePtr cast_dtype_;
  // std::shared_ptr<DeviceEvent> device_event_{nullptr};
  // UserData user_data_;
  mindspore::TensorCompressionType compression_type_;
  std::vector<QuantizationParamPtr> quant_params_;
  std::string tensor_name_;
};

class MapTensorData : public TensorData {
 public:
  MapTensorData(PyObject *obj, bool needSpecialize, int recurseDepth) : TensorData(obj, needSpecialize, recurseDepth) {
    tp_ = ItemType::MapTensor;
    needSpecialize = specialized_;
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    key_dtype_ = tensor_ptr->key_dtype();
    if (tensor_ptr->key_tensor() != nullptr) {
      key_shape_ = tensor_ptr->key_tensor()->shape();
    }
    default_value_ = tensor_ptr->default_value() == nullptr ? nullptr : tensor_ptr->default_value()->type()->Clone();
    permit_filter_value_ =
      tensor_ptr->permit_filter_value() == nullptr ? nullptr : tensor_ptr->permit_filter_value()->type()->Clone();
    evict_filter_value_ =
      tensor_ptr->evict_filter_value() == nullptr ? nullptr : tensor_ptr->evict_filter_value()->type()->Clone();
    value_shape_ = tensor_ptr->value_shape();
    key_tensor_ptr_ = tensor_ptr->key_tensor();
    value_tensor_ptr_ = tensor_ptr->value_tensor();
    status_tensor_ptr_ = tensor_ptr->status_tensor();
    if (recurseDepth > 0) {
      key_tensor_ = std::make_shared<TensorData>(key_tensor_ptr_, needSpecialize, recurseDepth);
      value_tensor_ = std::make_shared<TensorData>(value_tensor_ptr_, needSpecialize, recurseDepth);
      status_tensor_ = std::make_shared<TensorData>(status_tensor_ptr_, needSpecialize, recurseDepth);
    }
  }

  bool IsPermitFilterValue(const MapTensorData &other) {
    return (other.default_value_ == nullptr && default_value_ == nullptr) ||
           (other.default_value_ != nullptr && default_value_ != nullptr && *default_value_ == *(other.default_value_));
  }

  bool IsDefaultValue(const MapTensorData &other) {
    return (other.default_value_ == nullptr && default_value_ == nullptr) ||
           (other.default_value_ != nullptr && default_value_ != nullptr && *default_value_ == *(other.default_value_));
  }

  bool IsEvictFilterValue(const MapTensorData &other) {
    return (other.evict_filter_value_ == nullptr && evict_filter_value_ == nullptr) ||
           (other.evict_filter_value_ != nullptr && evict_filter_value_ != nullptr &&
            *evict_filter_value_ == *(other.evict_filter_value_));
  }

  bool operator==(const ItemData &obj) override {
    if (!ItemData::operator==(obj)) {
      return false;
    }
    const MapTensorData &other = (const MapTensorData &)obj;
    bool ret = TensorData::operator==(obj);
    return ret && other.key_dtype_ == key_dtype_ && other.key_shape_ == key_shape_ && IsDefaultValue(other) &&
           IsPermitFilterValue(other) && IsEvictFilterValue(other) && value_shape_ == other.value_shape_ &&
           ((recurseDepth_ > 0 && *key_tensor_ == *(other.key_tensor_) && *value_tensor_ == *(other.value_tensor_) &&
             *status_tensor_ == *(other.status_tensor_)) ||
            (recurseDepth_ <= 0 && ((key_tensor_ptr_ == nullptr) == (other.key_tensor_ptr_ == nullptr)) &&
             ((value_tensor_ptr_ == nullptr) == (other.value_tensor_ptr_ == nullptr)) &&
             ((status_tensor_ptr_ == nullptr) == (other.status_tensor_ptr_ == nullptr))));
  }

  std::string ToString() override {
    std::string map_tensor = ToStringIntern();
    return DESC(map_tensor) + DESC_END;
  }

 protected:
  std::string ToStringIntern() override {
    return TensorData::ToStringIntern() + DESC_STRING(key_dtype_) + DESC_TOSTRING(default_value_) +
           DESC_TOSTRING(permit_filter_value_) + DESC_TOSTRING(evict_filter_value_) + DESC_TOSTRING(key_tensor_) +
           DESC_TOSTRING(value_tensor_) + DESC_TOSTRING(status_tensor_) + DESC_END;
  }

  mindspore::TypeId key_dtype_;
  ShapeVector key_shape_;
  TypePtr default_value_;
  TypePtr permit_filter_value_;
  TypePtr evict_filter_value_;
  ShapeVector value_shape_;
  std::shared_ptr<TensorData> key_tensor_;
  std::shared_ptr<TensorData> value_tensor_;
  std::shared_ptr<TensorData> status_tensor_;
  mindspore::tensor::TensorPtr key_tensor_ptr_;
  mindspore::tensor::TensorPtr value_tensor_ptr_;
  mindspore::tensor::TensorPtr status_tensor_ptr_;
};

class RowTensorData : public ItemData {
 public:
  RowTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::RowTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::RowTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ptr_ = tensor_ptr->GetIndices();
    values_ptr_ = tensor_ptr->GetValues();
    if (recurseDepth > 0) {
      indices_ = std::make_shared<TensorData>(indices_ptr_, needSpecialize, recurseDepth);
      values_ = std::make_shared<TensorData>(values_ptr_, needSpecialize, recurseDepth);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const RowTensorData &other = (const RowTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             ((recurseDepth_ > 0 && *indices_ == *(other.indices_) && *values_ == *(other.values_)) ||
              (recurseDepth_ <= 0 && (indices_ptr_ == nullptr) == (other.indices_ptr_ == nullptr) &&
               (values_ptr_ == nullptr) == (other.values_ptr_ == nullptr)));
    }
    return false;
  }

  std::string ToString() override {
    std::string row_tensor = DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_STRING(data_type_);
    return DESC(row_tensor) + DESC_END;
  }

 protected:
  mindspore::tensor::TensorPtr indices_ptr_;
  mindspore::tensor::TensorPtr values_ptr_;
  std::shared_ptr<TensorData> indices_;
  std::shared_ptr<TensorData> values_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class COOTensorData : public ItemData {
 public:
  COOTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::COOTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::COOTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ptr_ = tensor_ptr->GetIndices();
    values_ptr_ = tensor_ptr->GetValues();
    if (recurseDepth > 0) {
      indices_ = std::make_shared<TensorData>(indices_ptr_, needSpecialize, recurseDepth);
      values_ = std::make_shared<TensorData>(values_ptr_, needSpecialize, recurseDepth);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const COOTensorData &other = (const COOTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             ((recurseDepth_ > 0 && *indices_ == *(other.indices_) && *values_ == *(other.values_)) ||
              (recurseDepth_ <= 0 && (indices_ptr_ == nullptr) == (other.indices_ptr_ == nullptr) &&
               (values_ptr_ == nullptr) == (other.values_ptr_ == nullptr)));
    }
    return false;
  }

  std::string ToString() override {
    std::string coo_tensor = DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_STRING(data_type_);
    return DESC(coo_tensor) + DESC_END;
  }

 protected:
  mindspore::tensor::TensorPtr indices_ptr_;
  mindspore::tensor::TensorPtr values_ptr_;
  std::shared_ptr<TensorData> indices_;
  std::shared_ptr<TensorData> values_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class CSRTensorData : public ItemData {
 public:
  CSRTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::CSRTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::CSRTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ptr_ = tensor_ptr->GetIndices();
    values_ptr_ = tensor_ptr->GetValues();
    indptr_ptr_ = tensor_ptr->GetIndptr();
    if (recurseDepth > 0) {
      indices_ = std::make_shared<TensorData>(indices_ptr_, needSpecialize, recurseDepth);
      values_ = std::make_shared<TensorData>(values_ptr_, needSpecialize, recurseDepth);
      indptr_ = std::make_shared<TensorData>(indptr_ptr_, needSpecialize, recurseDepth);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const CSRTensorData &other = (const CSRTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             ((recurseDepth_ > 0 && *indices_ == *(other.indices_) && *values_ == *(other.values_) &&
               *indptr_ == *(other.indptr_)) ||
              (recurseDepth_ <= 0 && indices_ptr_ == other.indices_ptr_ && values_ptr_ == other.values_ptr_ &&
               indptr_ptr_ == other.indptr_ptr_));
    }
    return false;
  }

  std::string ToString() override {
    std::string csr_tensor =
      DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_TOSTRING(indptr_) + DESC_STRING(data_type_);
    return DESC(csr_tensor) + DESC_END;
  }

 protected:
  mindspore::tensor::TensorPtr indices_ptr_;
  mindspore::tensor::TensorPtr values_ptr_;
  mindspore::tensor::TensorPtr indptr_ptr_;
  std::shared_ptr<TensorData> indices_;
  std::shared_ptr<TensorData> values_;
  std::shared_ptr<TensorData> indptr_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class TensorDataData : public ItemData {
 public:
  TensorDataData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Tensordata, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    data_ = pyObj.cast<mindspore::tensor::TensorDataPtr>();
    size_ = data_->size();
    itemsize_ = data_->itemsize();
    nbytes_ = data_->nbytes();
    ndim_ = data_->ndim();
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const TensorDataData &other = (const TensorDataData &)obj;
      if (specialized_) {
        return data_->equals(*(other.data_));
      } else {
        return size_ == other.size_ && itemsize_ == other.itemsize_ && nbytes_ == other.nbytes_ &&
               ndim_ == other.ndim_ && (data_ == nullptr) == (other.data_ == nullptr);
      }
    }
    return false;
  }

  std::string ToString() override {
    std::string tensor_data = DESC_STRING(size_) + DESC_STRING(itemsize_) + DESC_STRING(nbytes_) + DESC_STRING(ndim_);
    return DESC(tensor_data) + DESC_END;
  }

 protected:
  mindspore::tensor::TensorDataPtr data_;
  ssize_t size_;
  ssize_t itemsize_;
  ssize_t nbytes_;
  ssize_t ndim_;
};

class PrimitiveData : public ItemData {
 public:
  PrimitiveData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Primitive, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto data = pyObj.cast<PrimitivePyAdapterPtr>();
    py::dict pd = data->GetAttrDict();
    auto dct = pd.ptr();
    Py_ssize_t pos = 0;
    PyObject *key, *val;
    while (PyDict_Next(dct, &pos, &key, &val)) {
      ItemDataPtr k, v;
      if (recurseDepth > 0 || needSpecialize) {
        k = CreateItem(key, needSpecialize, recurseDepth);
        v = CreateItem(val, needSpecialize, recurseDepth);
      } else {
        k =
          CreateItem((key == NULL || key == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(key)), false, false);
        v =
          CreateItem((val == NULL || val == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(val)), false, false);
      }
      listK_.push_back(k);
      listV_.push_back(v);
    }
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const PrimitiveData &other = (const PrimitiveData &)obj;
      if (other.listK_.size() == listK_.size() && other.listV_.size() == listV_.size()) {
        for (size_t i = 0; i < listK_.size(); ++i) {
          if (*(listK_[i]) == *(other.listK_[i]) && *(listV_[i]) == *(other.listV_[i])) {
            continue;
          } else {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  std::string ToString() override {
    std::string primitive;
    for (size_t i = 0; i < listK_.size(); ++i) {
      primitive += DESC_ITEM(listK_[i], listV_[i]);
    }
    return DESC(primitive) + DESC_END;
  }

 protected:
  std::vector<ItemDataPtr> listK_;
  std::vector<ItemDataPtr> listV_;
};

class CellData : public ItemData {
 public:
  CellData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Cell, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto cell = pyObj.cast<mindspore::CellPtr>();
    PyObject *ns = PyObject_GetAttrString(obj, "__dict__");
    if (!ns) {
      return;
    }
    PyObject *items = PyMapping_Items(ns);
    if (!items) {
      return;
    }
    for (Py_ssize_t pos = 0; pos < PyList_GET_SIZE(items); pos++) {
      PyObject *it = PySequence_Fast(PyList_GET_ITEM(items, pos), "items() returned non-iterable");
      if (!it || PySequence_Fast_GET_SIZE(it) != 2) {
        if (it) {
          Py_DECREF(it);
        }
        continue;
      }
      PyObject *key = PySequence_Fast_GET_ITEM(it, 0);
      PyObject *val = PySequence_Fast_GET_ITEM(it, 1);
      ItemDataPtr k, v;
      if (recurseDepth > 0 || needSpecialize) {
        k = CreateItem(key, needSpecialize, recurseDepth);
        v = CreateItem(val, needSpecialize, recurseDepth);
      } else {
        k =
          CreateItem((key == NULL || key == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(key)), false, false);
        v =
          CreateItem((val == NULL || val == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(val)), false, false);
      }
      listK_.push_back(k);
      listV_.push_back(v);
      Py_DECREF(it);
    }
    Py_DECREF(items);
    Py_DECREF(ns);
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      const CellData &other = (const CellData &)obj;
      for (size_t i = 0; i < listK_.size(); ++i) {
        if (*(listK_[i]) == *(other.listK_[i]) && *(listV_[i]) == *(other.listV_[i])) {
          continue;
        } else {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  std::string ToString() override {
    std::string cell;
    for (size_t i = 0; i < listK_.size(); ++i) {
      cell += DESC_ITEM(listK_[i], listV_[i]);
    }
    return DESC(cell) + DESC_END;
  }

 protected:
  std::vector<ItemDataPtr> listK_;
  std::vector<ItemDataPtr> listV_;
};

class UnknownData : public ItemData {
 public:
  UnknownData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyUnknown, needSpecialize, recurseDepth) {
    refId_ = obj;
  }

  bool operator==(const ItemData &obj) override {
    if (ItemData::operator==(obj)) {
      return refId_ == ((const UnknownData &)obj).refId_;
    }
    return false;
  }

  std::string ToString() override {
    std::string ret = "unknown";
    return ret + ItemData::ToString();
  }

 protected:
  PyObject *refId_;
};

using CheckPyObjectFunc = bool (*)(PyObject *obj);
using CreatePyObjectFunc = ItemDataPtr (*)(PyObject *obj, bool need_specialize, int recurse_depth);
template <typename T>
ItemDataPtr CreatePyData(PyObject *obj, bool need_specialize, int recurse_depth) {
  return std::make_shared<T>(obj, need_specialize, recurse_depth);
}
template <typename T>
ItemDataPtr CreateMutablePyData(PyObject *obj, bool need_specialize, int recurse_depth) {
  return std::make_shared<T>(obj, false, recurse_depth);
}
static const std::vector<std::pair<CheckPyObjectFunc, CreatePyObjectFunc>> kFuncPyObjectConverter = {
  {[](PyObject *obj) -> bool { return PyLong_Check(obj) && !PyBool_Check(obj); }, CreatePyData<IntData>},
  {[](PyObject *obj) -> bool { return !!PyFloat_Check(obj); }, CreatePyData<FloatData>},
  {[](PyObject *obj) -> bool { return !!PyBool_Check(obj); }, CreatePyData<BoolData>},
  {[](PyObject *obj) -> bool { return !!PyBytes_Check(obj); }, CreatePyData<BytesData>},
  {[](PyObject *obj) -> bool { return !!PyUnicode_Check(obj); }, CreatePyData<StringData>},
  {[](PyObject *obj) -> bool { return !!PyList_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PyTuple_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PySet_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PyFrozenSet_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PyDict_Check(obj); }, CreatePyData<DictData>},
  {[](PyObject *obj) -> bool { return !!PyComplex_Check(obj); }, CreatePyData<ComplexData>},
  {[](PyObject *obj) -> bool { return !!PySlice_Check(obj); }, CreatePyData<SliceData>},
  {[](PyObject *obj) -> bool { return !!PyFunction_Check(obj); }, CreatePyData<FunctionData>},
  {[](PyObject *obj) -> bool { return !!PyMethod_Check(obj); }, CreatePyData<MethodData>},
  {[](PyObject *obj) -> bool { return !!PyInstanceMethod_Check(obj); }, CreatePyData<InstanceMethodData>},
  {[](PyObject *obj) -> bool { return !!PyType_Check(obj); }, CreatePyData<TypeData>},
  {[](PyObject *obj) -> bool { return py::isinstance<py::array>(obj); }, CreatePyData<NumpyData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::Type>(obj); }, CreatePyData<TensorTypeData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::MapTensor>(obj); },
   CreatePyData<MapTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::Tensor>(obj); }, CreatePyData<TensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::ParamInfo>(obj); }, CreatePyData<ParamInfoData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::MetaTensor>(obj); },
   CreatePyData<MetaTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::TensorData>(obj); },
   CreatePyData<TensorDataData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::PrimitivePyAdapter>(obj); },
   CreatePyData<PrimitiveData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::Cell>(obj); }, CreatePyData<CellData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::RowTensor>(obj); },
   CreateMutablePyData<RowTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::COOTensor>(obj); },
   CreateMutablePyData<COOTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::CSRTensor>(obj); },
   CreateMutablePyData<CSRTensorData>},
};

static ItemDataPtr CreateData(PyObject *obj, bool need_specialize, int recurse_depth) {
  auto tar =
    std::find_if(kFuncPyObjectConverter.begin(), kFuncPyObjectConverter.end(),
                 [obj](const std::pair<CheckPyObjectFunc, CreatePyObjectFunc> &func) { return func.first(obj); });
  if (tar != kFuncPyObjectConverter.end()) {
    return tar->second(obj, need_specialize, recurse_depth);
  } else {
    return std::make_shared<UnknownData>(obj, need_specialize, recurse_depth);
  }
}

static ItemDataPtr CreateItem(PyObject *obj, bool need_specialize, int recurse_depth) {
  ReprRecursionScope scope(obj);
  if (scope.ReEnterOrError()) {
    return std::make_shared<ItemData>(ItemType::PyNull, need_specialize, recurse_depth);
  }
  if (recurse_depth < -1) {
    if (obj != NULL && obj != Py_None) {
      py::object py_obj = py::reinterpret_borrow<py::object>(obj);
      if (IsStubTensor(py_obj)) {
        py_obj = python_adapter::CallPyObjMethod(py_obj, "stub_sync");
        obj = py_obj.ptr();
      }
      return std::make_shared<TypeData>(reinterpret_cast<PyObject *>(Py_TYPE(obj)), false, 0);
    } else {
      return std::make_shared<ItemData>(ItemType::PyNull, false, 0);
    }
  }
  recurse_depth -= 1;
  ItemDataPtr dp;
  if (obj != NULL && obj != Py_None) {
    py::object py_obj = py::reinterpret_borrow<py::object>(obj);
    if (IsStubTensor(py_obj)) {
      py_obj = python_adapter::CallPyObjMethod(py_obj, "stub_sync");
      obj = py_obj.ptr();
    }
    dp = CreateData(obj, need_specialize, recurse_depth);
  } else {
    dp = std::make_shared<ItemData>(ItemType::PyNull, need_specialize, recurse_depth);
  }
  return dp;
}

GuardItem::GuardItem(TracePtr tt) : var_(tt) {}

void GuardItem::Replace(TracePtr dst, TracePtr src) {
  if (!var_) {
    return;
  }
  if (*var_ == *src) {
    var_ = dst;
  } else {
    var_->Replace(dst, src);
  }
}

class EqGuard : public GuardItem {
 public:
  EqGuard(TracePtr obj, bool needSpecialize, int recurseDepth)
      : GuardItem(obj),
        dp_(CreateItem(obj->GetObject(), needSpecialize, recurseDepth)),
        specialized_(needSpecialize),
        recurse_(recurseDepth) {}

  virtual bool Check(PyFrameObject *frame) {
    PyObject *obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    ItemDataPtr other = CreateItem(obj, specialized_, recurse_);
    return *dp_ == *other;
  }

  virtual std::string ToString() { return var_->ToString() + "==" + dp_->ToString(); }

 protected:
  ItemDataPtr dp_;
  bool specialized_;
  int recurse_;
};

class TypeGuard : public GuardItem {
 public:
  explicit TypeGuard(TracePtr obj) : GuardItem(obj) {
    if (obj->GetTraceType() == TraceType::Type) {
      refType_ = std::dynamic_pointer_cast<TypeTrace>(obj)->GetType();
    } else {
      refType_ = Py_TYPE(obj->GetObject());
    }
  }

  virtual bool Check(PyFrameObject *frame) {
    PyObject *obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj);
    if (var_->GetTraceType() != TraceType::Type && obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    if (obj == NULL) {
      return false;
    }
    PyTypeObject *tp;
    if (var_->GetTraceType() == TraceType::Type) {
      tp = reinterpret_cast<PyTypeObject *>(obj);
    } else {
      tp = Py_TYPE(obj);
    }
    if (tp != refType_) {
      return false;
    } else {
      return true;
    }
  }

  std::string ToString() override {
    if (var_->GetTraceType() == TraceType::Type) {
      return var_->ToString() + std::string("==") + refType_->tp_name;
    } else {
      return std::string("type(") + var_->ToString() + std::string(")==") + refType_->tp_name;
    }
  }

 protected:
  PyTypeObject *refType_;
};

class IdGuard : public GuardItem {
 public:
  explicit IdGuard(TracePtr obj) : GuardItem(obj) { refId_ = obj->GetObject(); }

  virtual bool Check(PyFrameObject *frame) {
    PyObject *obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    bool ret = false;
    if (obj == NULL) {
      return ret;
    }
    if (obj != refId_) {
      ret = false;
    } else {
      ret = true;
    }
    return ret;
  }

  std::string ToString() override {
    return std::string("id(") + var_->ToString() + std::string(")==") + std::to_string((size_t)refId_);
  }

 protected:
  PyObject *refId_;
};

class AttrGuard : public GuardItem {
 public:
  explicit AttrGuard(TracePtr pObj) : GuardItem(pObj) {
    AttrTracePtr t = std::dynamic_pointer_cast<AttrTrace>(pObj);
    PyObject *obj = t->GetOrigin()->GetObject();
    nameAttr_ = t->GetAttribute();
    if (PyObject_HasAttrString(obj, nameAttr_.c_str()) != 0) {
      hasAttr_ = true;
    } else {
      hasAttr_ = false;
      bool is_dict = PyDict_CheckExact(obj);
      PyObject *itemName = PyUnicode_FromString(nameAttr_.c_str());
      PyObject *attr = NULL;
      if (is_dict) {
        attr = PyDict_GetItem(obj, itemName);
        if (attr != NULL) {
          Py_INCREF(attr);
        }
      } else if (PyMapping_Check(obj) || PySequence_Check(obj)) {
        attr = PyObject_GetItem(obj, itemName);
      }
      hasAttr_ = attr != NULL;
      Py_DECREF(itemName);
      if (attr != NULL) {
        Py_DECREF(attr);
      }
    }
  }

  virtual ~AttrGuard() {}

  virtual bool Check(PyFrameObject *frame) {
    PyObject *obj = GetObjectFromTrace(frame, var_);
    bool ret = CheckIntern(obj);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    bool ret;
    if (PyObject_HasAttrString(obj, nameAttr_.c_str()) != 0) {
      ret = hasAttr_;
    } else {
      bool is_dict = PyDict_CheckExact(obj);
      PyObject *itemName = PyUnicode_FromString(nameAttr_.c_str());
      PyObject *attr = NULL;
      if (is_dict) {
        attr = PyDict_GetItem(obj, itemName);
        if (attr != NULL) {
          Py_INCREF(attr);
        }
      } else if (PyMapping_Check(obj) || PySequence_Check(obj)) {
        attr = PyObject_GetItem(obj, itemName);
      }
      ret = CheckIntern(attr);
      Py_DECREF(itemName);
      if (attr != NULL) {
        Py_DECREF(attr);
      }
    }
    return ret;
  }

  virtual bool CheckIntern(PyObject *obj) {
    bool ret;
    if ((obj == NULL && !hasAttr_) || (obj != NULL && hasAttr_)) {
      ret = true;
    } else {
      ret = false;
    }
    return ret;
  }

  virtual std::string ToString() { return std::string("exist(") + var_->ToString() + std::string(")"); }

 protected:
  bool hasAttr_;
  std::string nameAttr_;
};

GuardItemPtr GuardEqual(TracePtr obj, bool needSpecialize, int recurseDepth) {
  return std::make_shared<EqGuard>(obj, needSpecialize, recurseDepth);
}

GuardItemPtr GuardType(TracePtr obj) { return std::make_shared<TypeGuard>(obj); }

GuardItemPtr GuardId(TracePtr obj) { return std::make_shared<IdGuard>(obj); }

GuardItemPtr GuardAttr(TracePtr obj) {
  if (obj->GetTraceType() != TraceType::Attr) {
    return nullptr;
  } else {
    return std::make_shared<AttrGuard>(obj);
  }
}

bool IsPyObjectEqual(PyObject *src, PyObject *dst) {
  if (src == dst) {
    return true;
  }
  ItemDataPtr src_item = CreateItem(src, true, INT_MAX);
  ItemDataPtr dst_item = CreateItem(dst, true, INT_MAX);
  return *src_item == *dst_item;
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
