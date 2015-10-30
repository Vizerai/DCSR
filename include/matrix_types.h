#ifndef MATRIX_TYPES_H
#define MATRIX_TYPES_H

#define NUM_MATRIX_TYPES	27

enum MatrixType
{
	r,
    r_prime,
    sigma,
    sigma_prime,
    CondTrue,
    CondFalse,
    Body,
    Fun,
    Arg,
    Var,
    Call,
    temp_Mat,
    a,
    a_var,
    a_set,
    v,
    vf,
    v_set,
    void_vec,
    not_false_vec,
    false_vec,
    PrimBool,
    PrimNum,
    PrimVoid,
    PrimList,
    IF,
    SET
};

const char* const MatrixTypeMap[] =
{
	"r",
    "r_prime",
    "sigma",
    "sigma_prime",
    "CondTrue",
    "CondFalse",
    "Body",
    "Fun",
    "Arg",
    "Var",
    "Call",
    "temp_Mat",
    "a",
    "a_var",
    "a_set",
    "v",
    "vf",
    "v_set",
    "void_vec",
    "not_false_vec",
    "false_vec",
    "PrimBool",
    "PrimNum",
    "PrimVoid",
    "PrimList",
    "If",
    "Set"
};

#endif