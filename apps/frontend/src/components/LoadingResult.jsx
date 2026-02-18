// import "./LoadingResult.css"

const LoadingResult = (props) => {
  return (<>
    <div
      className={`
        d-flex justify-content-center align-items-center
        ${props.setClass}
      `}
      style={{color: "#A9A9A9", ...props.setStyle}}
    >
      <div
        id="loading-spinner"
        className="spinner-border"
        style={{
          height: "0.6em",
          width: "0.6em",
          borderWidth: "3px",
        }}
        role='status'
      >
      </div>

      <div id="loading-text" className={props.setTextClass}>
        {props.loadingText}
      </div>

    </div>
  </>)
};

export default LoadingResult;
