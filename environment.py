"""
Environment for executing and evaluating code.
"""

import sys
import ast
import time
import traceback
import multiprocessing
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Any, Optional, Union

# Define a helper function for the multiprocessing queue
# This needs to be at the module level to be picklable
def _execute_and_queue(exec_function, code, test_case, expected_output, queue):
    """Helper function to execute code and put results in a queue."""
    result = exec_function(code, test_case, expected_output)
    queue.put(result)

class CodeExecutionEnvironment:
    """Environment for executing and evaluating code."""
    
    def __init__(self, timeout: int = 5):
        """
        Initialize CodeExecutionEnvironment.
        
        Args:
            timeout: Timeout in seconds for code execution
        """
        self.timeout = timeout
    
    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code has valid syntax.
        
        Args:
            code: Code to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def execute_code(self, code: str, test_case: str, expected_output: Any = None) -> Dict[str, Any]:
        """
        Execute code with test case.
        
        Args:
            code: Code to execute
            test_case: Test case to run
            expected_output: Expected output
            
        Returns:
            Dictionary with results
        """
        # Create namespace for execution
        namespace = {}
        
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Result dictionary
        result = {
            "success": False,
            "compile_error": None,
            "runtime_error": None,
            "output": None,
            "expected": expected_output,
            "execution_time": None,
            "test_case": test_case,
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "match": False
        }
        
        try:
            # First check syntax
            compile(code, "<string>", "exec")
            
            # Execute code to define functions/classes
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            # Now execute test case
            start_time = time.time()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Check if test_case is an assert statement
                if test_case.strip().startswith("assert "):
                    # For assert statements, we need to use exec
                    try:
                        exec(test_case, namespace)
                        test_result = True  # If assertion passed
                    except AssertionError:
                        test_result = False  # If assertion failed
                    except Exception as e:
                        raise e  # Re-raise other exceptions
                else:
                    # For expressions, we can use eval
                    test_result = eval(test_case, namespace)
            
            execution_time = time.time() - start_time
            
            # Capture stdout and stderr
            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()
            result["output"] = test_result
            result["execution_time"] = execution_time
            
            # Check if output matches expected output
            if expected_output is not None:
                result["match"] = test_result == expected_output
                result["success"] = result["match"]
            else:
                # If no expected output, we assume success if no errors
                result["success"] = True
                result["match"] = True
            
        except SyntaxError as e:
            result["compile_error"] = str(e)
            result["traceback"] = traceback.format_exc()
        except Exception as e:
            result["runtime_error"] = str(e)
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def execute_test_case(self, code: str, test_case: str, expected_output: Any = None) -> Dict[str, Any]:
        """
        Execute test case with timeout.
        
        Args:
            code: Code to execute
            test_case: Test case to run
            expected_output: Expected output
            
        Returns:
            Dictionary with results
        """
        # Use multiprocessing to handle timeout
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        
        # Create process with the helper function instead of lambda
        p = ctx.Process(
            target=_execute_and_queue,
            args=(self.execute_code, code, test_case, expected_output, q)
        )
        
        # Start process and wait for timeout
        p.start()
        p.join(self.timeout)
        
        # Check if process timed out
        if p.is_alive():
            p.terminate()
            p.join()
            
            return {
                "success": False,
                "compile_error": None,
                "runtime_error": f"Timeout after {self.timeout} seconds",
                "output": None,
                "expected": expected_output,
                "execution_time": None,
                "test_case": test_case,
                "stdout": "",
                "stderr": "",
                "traceback": f"Process timed out after {self.timeout} seconds",
                "match": False
            }
        else:
            # Get result from queue
            try:
                result = q.get(block=False)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "compile_error": None,
                    "runtime_error": str(e),
                    "output": None,
                    "expected": expected_output,
                    "execution_time": None,
                    "test_case": test_case,
                    "stdout": "",
                    "stderr": "",
                    "traceback": traceback.format_exc(),
                    "match": False
                }
    
    def execute_mbpp_test_cases(self, code: str, test_cases: List[str]) -> Dict[str, Any]:
        """
        Execute MBPP test cases.
        
        Args:
            code: Code to execute
            test_cases: List of test cases
            
        Returns:
            Dictionary with results
        """
        # Check syntax first
        is_valid, error = self.check_syntax(code)
        if not is_valid:
            return {
                "success": False,
                "compile_error": error,
                "test_results": [],
                "pass_rate": 0.0
            }
        
        # Execute each test case
        test_results = []
        for test_case in test_cases:
            # Create a test script that runs the assertion and checks if it passes
            test_script = f"""
{code}

def run_test():
    try:
        {test_case}
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
"""
            # Run the test
            result = self.execute_test_case(test_script, "run_test()")
            result["test_case"] = test_case
            test_results.append(result)
        
        # Calculate pass rate
        passed = sum(1 for r in test_results if r["success"] and r["output"] is True)
        pass_rate = passed / len(test_results) if test_results else 0.0
        
        return {
            "success": pass_rate == 1.0,
            "compile_error": None,
            "test_results": test_results,
            "pass_rate": pass_rate
        }
    
    def execute_humaneval_test(self, code: str, test_code: str, entry_point: str) -> Dict[str, Any]:
        """
        Execute HumanEval test.
        
        Args:
            code: Code to execute
            test_code: Test code
            entry_point: Entry point function
            
        Returns:
            Dictionary with results
        """
        # Combine code and test code
        full_code = f"{code}\n\n{test_code}"
        
        # Check syntax
        is_valid, error = self.check_syntax(full_code)
        if not is_valid:
            return {
                "success": False,
                "compile_error": error,
                "output": None
            }
        
        # Execute test
        result = self.execute_test_case(full_code, "True")
        
        return {
            "success": result["success"] and not result["runtime_error"],
            "output": "All tests passed!" if (result["success"] and not result["runtime_error"]) else f"Test failed: {result['runtime_error']}",
            "compile_error": result.get("compile_error")
        }