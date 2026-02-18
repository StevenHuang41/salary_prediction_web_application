const ErrorPredict = ({ data }) => {

  return (<>
    <div
      className="row m-0 align-items-center text-bg-danger"
      style={{height: "15vh"}}
    >
      <div
        className="text-center fs-1"
      >
        {data}
      </div>
    </div>
  </>)

};

export default ErrorPredict;
