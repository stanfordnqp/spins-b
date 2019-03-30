from spins.invdes.problem_graph import optplan

import pytest


def test_constant_standard():
    """Tests `make_constant` creation using standard procedure."""
    constant = optplan.make_constant(
        value=optplan.ComplexNumber(real=2, imag=3.4))
    assert constant.value.real == 2
    assert constant.value.imag == 3.4


@pytest.mark.parametrize("value", [1, 1.0, 1 + 2j])
def test_constant_short(value):
    """Tests `make_constant` creation using the short form (e.g. `Constant(6)`)."""
    constant = optplan.make_constant(value)
    complex_value = complex(value)
    assert constant.value.real == complex_value.real
    assert constant.value.imag == complex_value.imag


def test_sum_add_to_sum():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    var3 = optplan.Parameter()
    var4 = optplan.Parameter()
    sum1 = optplan.Sum(functions=[var1, var2])
    sum2 = optplan.Sum(functions=[var3, var4])
    sum3 = sum1 + sum2

    assert isinstance(sum3, optplan.Sum)
    assert sum3.functions == [var1, var2, var3, var4]


def test_sum_add_to_fun():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    var3 = optplan.Parameter()
    sum1 = optplan.Sum(functions=[var1, var2])
    sum2 = sum1 + var3

    assert isinstance(sum2, optplan.Sum)
    assert sum2.functions == [var1, var2, var3]


def test_sum_add_to_value():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    sum1 = optplan.Sum(functions=[var1, var2])
    sum2 = sum1 + 3

    assert isinstance(sum2, optplan.Sum)
    assert var1 in sum2.functions
    assert var2 in sum2.functions
    assert sum2.functions[-1].value.real == 3
    assert sum2.functions[-1].value.imag == 0


def test_sum_add_bad_node_raise_type_error():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    sum1 = optplan.Sum(functions=[var1, var2])

    with pytest.raises(TypeError, match="add a node"):
        sum1 + optplan.SimulationSpace()


def test_prod_mul_to_prod():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    var3 = optplan.Parameter()
    var4 = optplan.Parameter()
    prod1 = optplan.Product(functions=[var1, var2])
    prod2 = optplan.Product(functions=[var3, var4])
    prod3 = prod1 * prod2

    assert isinstance(prod3, optplan.Product)
    assert prod3.functions == [var1, var2, var3, var4]


def test_prod_mul_to_fun():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    var3 = optplan.Parameter()
    prod1 = optplan.Product(functions=[var1, var2])
    prod2 = prod1 * var3

    assert isinstance(prod2, optplan.Product)
    assert prod2.functions == [var1, var2, var3]


def test_prod_mul_to_value():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    prod1 = optplan.Product(functions=[var1, var2])
    prod2 = prod1 * 3

    assert isinstance(prod2, optplan.Product)
    assert var1 in prod2.functions
    assert var2 in prod2.functions
    assert prod2.functions[-1].value.real == 3
    assert prod2.functions[-1].value.imag == 0


def test_prod_mul_bad_node_raise_type_error():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    prod1 = optplan.Product(functions=[var1, var2])

    with pytest.raises(TypeError, match="multiply a node"):
        prod1 * optplan.SimulationSpace()


def test_sum_two_funs():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    sum1 = var1 + var2

    assert isinstance(sum1, optplan.Sum)
    assert sum1.functions == [var1, var2]


def test_sum_fun_and_const():
    var1 = optplan.Parameter()
    sum1 = var1 + 2
    assert isinstance(sum1, optplan.Sum)
    assert len(sum1.functions) == 2
    assert sum1.functions[0] == var1
    assert sum1.functions[1].value.real == 2
    assert sum1.functions[1].value.imag == 0


def test_sum_fun_and_const_reverse():
    var1 = optplan.Parameter()
    sum1 = 2 + var1
    assert isinstance(sum1, optplan.Sum)
    assert len(sum1.functions) == 2
    assert sum1.functions[0] == var1
    assert sum1.functions[1].value.real == 2
    assert sum1.functions[1].value.imag == 0


def test_sum_fun_and_const_obj():
    var1 = optplan.Parameter()
    const1 = optplan.make_constant(2)
    sum1 = var1 + const1
    assert isinstance(sum1, optplan.Sum)
    assert len(sum1.functions) == 2
    assert sum1.functions[0] == var1
    assert sum1.functions[1] == const1


def test_sum_fun_and_bad_obj_raises_type_error():
    with pytest.raises(TypeError, match="add node"):
        optplan.Parameter() + optplan.SimulationSpace()


def test_prod_two_funs():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    prod1 = var1 * var2

    assert isinstance(prod1, optplan.Product)
    assert prod1.functions == [var1, var2]


def test_prod_fun_and_const():
    var1 = optplan.Parameter()
    prod1 = var1 * 2
    assert isinstance(prod1, optplan.Product)
    assert len(prod1.functions) == 2
    assert prod1.functions[0] == var1
    assert prod1.functions[1].value.real == 2
    assert prod1.functions[1].value.imag == 0


def test_prod_fun_and_const_reverse():
    var1 = optplan.Parameter()
    prod1 = 2 * var1
    assert isinstance(prod1, optplan.Product)
    assert len(prod1.functions) == 2
    assert prod1.functions[0] == var1
    assert prod1.functions[1].value.real == 2
    assert prod1.functions[1].value.imag == 0


def test_prod_fun_and_const_obj():
    var1 = optplan.Parameter()
    const1 = optplan.make_constant(2)
    prod1 = var1 * const1
    assert isinstance(prod1, optplan.Product)
    assert len(prod1.functions) == 2
    assert prod1.functions[0] == var1
    assert prod1.functions[1] == const1


def test_prod_fun_and_bad_obj_raises_type_error():
    with pytest.raises(TypeError, match="multiply node"):
        optplan.Parameter() * optplan.SimulationSpace()


def test_power_number():
    var1 = optplan.Parameter()
    power1 = var1**2
    assert isinstance(power1, optplan.Power)
    assert power1.function == var1
    assert power1.exp == 2


def test_power_constant():
    var1 = optplan.Parameter()
    power1 = var1**optplan.make_constant(2)
    assert isinstance(power1, optplan.Power)
    assert power1.function == var1
    assert power1.exp == 2


def test_sub():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    diff = var2 - var1
    assert isinstance(diff, optplan.Sum)


def test_div():
    var1 = optplan.Parameter()
    var2 = optplan.Parameter()
    div = var2 / var1
    assert isinstance(div, optplan.Product)
